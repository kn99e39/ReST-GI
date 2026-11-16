# WL97 audit, track E2: independent reproduction of P0-2 and P0-3 by the relation-sampling route

Date: 2026-11-16. Auditor: independent agent (track E2; a first attempt on this track was interrupted by a usage limit, this report
continues from its scripts and outputs after re-checking them, see section 8). Repository HEAD efcf86a, branch `Selective_Recompute`.
No existing file was modified; nothing was trained or tuned; nothing was committed.

## 1. Scope and method

Claims under audit (wording of the audit brief):

- **P0-2** "The M64 / 64-bin sender introduces systematic sharp-delivery bias that is not eliminated by relation sampling."
- **P0-3** "Continuous exact-direction and hit-point ER evaluation substantially closes the CK2 delivery error."

Original evidence: WL95A section B2 (`docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:68-86`, interpretation
`:123-132`, `:145-155`, verdict `:167-171`), code `experiments/ecro_attr95a/a95.py`, artifacts `~/rest_x95a/results/{an.json,deliv.npz,main.json}`
(local copy `results/ecro_attr95a/`).

Workload (identical to the original; this is the whole scope of both verdicts): scene CK2 variant `none`, **one point light, one static
geometry**; light that is scattered exactly twice, first on the sharp ER panel S1 (42 patches 1088..1129, material 1) and then on the
Lambert box A (54 patches 1130..1183, albedo 0.5); quantity = per-patch exitance of box A (channels summed); 64 = 4 x 16 equal-solid-angle
bins; K = 512 state rays per receiver patch (8 per bin, weight cos x 2 pi / 64 / 8).

Method. I wrote my own chain (`experiments/audit97/E2/e2_chain.py`, not importing `a95.py` / `an94.py`): relation drawing, ray tracing
wrappers, hit patch and outgoing bin at the hit, source encoding, table lookup, exact per-ray ER evaluation (own shadow test), the
Lambert receiver, exitance and all metrics. Borrowed primitives only: scene / patch tables (`scene86.Geom` through `mat94.ctx_ck2`),
Mitsuba ray intersection, the ER closure (`closure83_dr.DrClosure83`), and the object under test (the M64 matrix `A4` of
`ops95a.npz`, equal to `mat94.mq_ext()[1]` to 1.7e-6). In addition to the route asked for, I built two things the original did not have:

1. an **independent area-form reference** (panel area x receiver area sampling of the two-point transport integral, 2.7e8 samples
   per patch, relative SE of the total 1.6e-4), so that the path-traced reference `CKF_none e2_p` is itself tested;
2. a **2 x 2 factorial** of the sender projection (outgoing direction exact / uniform in its bin) x (position exact hit point / uniform in
   the hit patch), evaluated per ray on the same draws, because the original chain q256 -> X1c -> X1 changes two things at its first step.

Random draws: 2,048 draws with offsets shared by all receiver patches (as the deployed state does; seeds 9720000 + d) and 2,048 draws
with independent offsets per patch (seeds 9740000 + d); 4,096 in total, all different from the original (950000 + d) and from the deployed
realization (86001), which is evaluated separately. Two further independent sets (768 and 512 draws) back the supplementary runs.

Metrics (own code): ratio = sum(arm) / sum(ref); abs rel L1 = sum|arm - ref| / sum(ref); corr = Pearson over the 54 patches. "vs CKF" uses
the original reference (mean of the two halves, weights W_T); "vs area" uses my area-form reference. SE of pooled values = standard
error over draws (ratio) or bootstrap over draws (L1).

## 2. What was read and run

Read: the WL95A worklog; `scene86.py`, `pt86.py`, `pt86a.py`, `quad87.py`, `model86.py`, parts of `mat94.py`, `closure83_dr.py`, `a95.py`
(to learn the definitions being reproduced; not imported or copied). Server: LabServer63, private copy of the frozen code at
`~/rest_x97/E2/code` (commit c152120, copied from `~/rest_x95a/code`), all outputs under `~/rest_x97/E2/`. Total GPU use about 15 min
(peak 2.7 GB), of which about 9 min was a badly vectorised job that I stopped and rewrote (section 8).

Commands are listed in section 9.

## 3. Sanity and identity checks (own chain)

| check | result | file |
|---|---|---|
| ER closure cosine convention on Lambert states (materials 3, 4) | `eval(a, b)` = rho/pi x cos of the **second** argument to 1.8e-7; using the first argument is wrong by up to 96x | `checks.json: cos_convention_lambert_mat*` |
| ER reciprocity of f = eval / cos(second) on S1 | 7e-7 | `checks.json: S1_reciprocity_max_rel` |
| own bins vs `pt86.Ctx.bin_of` | agreement 1.0 on 4,096 directions | `checks.json` |
| once-scattered radiance of a Lambert panel under the point light vs rho/pi x I cos/d^2 | 1.7e-6 | `checks.json: lambert_panel_radiance_max_rel_dev` |
| **analytic Lambert form factor** (unit-radiance panel, Lambert's polygon formula, 9 panel-facing patches) | own hemisphere flight / analytic = 0.9993..1.0006 (total 0.99986); own area estimator / analytic = 0.9984..1.0022 (total 0.99989 +- 0.0002) | `checks.json: analytic_form_factor` |
| Lambert panel lit by the point light, delivery to patch 1134 | numpy quadrature 0.49856, area estimator 0.49864, hemisphere flight 0.49772 (256 draws); totals over 54 patches agree to 0.9998 | `checks.json: lambert_panel_delivery` |
| source encoding (16 fixed points) vs production `ctx.source` | rel L1 1.2e-6 | `senders.json: s16_vs_production` |
| light on the panel | all incident energy in ring 0 (cos 0.025..0.16, median 0.040), incident bins 12..15, no shadowing | `checks.json: panel_incidence_cos_range`, `mquad.json` |
| deployed relation (seed 86001) | offsets equal `ctx.u_state_np` bitwise; 4,376 production rays box A -> S1, 4,380 in my tracing; hit patch agrees on 99.98 %, bin on 100 % of the common rays. The 4 extra rays do not move any delivered value (next table), so they carry negligible weight; their cause (probably the ray-origin offset at grazing rays) was not traced | `checks.json: deployed_relation_identity` |

The incident geometry matters for every interpretation below: **the light reaches the panel at 81..89 degrees from its normal**, so the whole
experiment sits in the grazing ring (4.3 % of the training irradiance according to the worklog).

## 4. Results

### 4.1 Deployed realization (seed 86001): original arrays reproduced

| arm | original (ratio / abs rel L1) | own chain, same draw | rel L1 between the two 54-patch arrays |
|---|---|---|---|
| A4 = M64(s16), binned | 1.208 / 0.524 | 1.208 / 0.524 | 1.3e-6 |
| M64(s_pool) | 1.158 / 0.483 | 1.156 / 0.483 | 4.9e-3 (different pooled source points) |
| q (physical table) | 1.077 / 0.534 | 1.094 / 0.545 | 1.7e-2 (different q Monte Carlo) |
| X1c (exact direction, patch centre) | 0.719 / 0.388 | 0.719 / 0.388 | 1.9e-6 |
| X1 (exact direction, hit point) | 0.764 / 0.627 | 0.764 / 0.627 | 4.0e-6 |

The deterministic arms agree to 1e-6 with an independently written flight, receiver and metric. The original implementation of these
arms is therefore numerically correct (`analysis.json: deployed`). The deployed draw is the 10th percentile of my per-draw distribution of the
binned ratio (`summary_tables.json: deployed`), which confirms the worklog statement that the deployed state is a low draw.

### 4.2 Pooled draws: original vs own

Own values pool 4,096 draws (the shared and the independent sets agree with each other within their SE; both are in `analysis.json`).

| arm | original, 64 draws: ratio / L1 (SE) | own, 4,096 draws vs CKF: ratio (SE) / L1 | own vs area reference: ratio (SE) / L1 (SE) / corr | noise-only L1 of the pooled mean |
|---|---|---|---|---|
| A4 = M64(s16), binned | 1.544 / 0.546 (0.035) | 1.529 (0.004) / 0.529 | **1.506 (0.004) / 0.506 (0.004)** / 0.984 | 0.011 |
| M64(s_pool) | 1.495 / 0.500 (0.034) | 1.475 (0.004) / 0.476 | 1.453 (0.004) / 0.453 (0.004) / 0.981 | 0.011 |
| q (physical table, exact incident light) | 1.422 / 0.433 (0.033) | 1.420 (0.004) / 0.431 | 1.399 (0.004) / 0.408 (0.004) / 0.964 | 0.010 |
| X1c (exact direction, patch centre) | 1.037 / 0.268 (0.045) | 1.080 (0.008) / 0.306 | 1.064 (0.008) / 0.269 (0.008) / 0.941 | 0.017 |
| own: exact direction, position uniform in the hit patch | not in the original | 1.050 (0.005) / 0.145 | 1.034 (0.005) / 0.111 (0.005) / 0.993 | 0.013 |
| **X1 (exact direction, hit point)** | 1.022 / 0.111 (0.041) | 1.010 (0.009) / 0.084 | **0.994 (0.009) / 0.029 (0.007)** / 0.999 | 0.021 |

The original 64-draw values sit inside my distribution of 64-draw pools (`summary_tables.json: wl95a_arrays`): A4 ratio z = +0.47
(own 64-draw pools 1.529 +- 0.032); q z = +0.05; X1c z = -0.57; X1 ratio z = +0.21. One point of caution: the original X1 map error
0.111 is the 3rd percentile of my 64-draw pools (0.174 +- 0.044, range 0.092..0.284). The original number was a favourable pool; its
conclusion nevertheless holds, because with 4,096 draws X1 reaches 0.029 against the area reference.

### 4.3 Does more relation sampling reduce the error? (mean abs rel L1 vs the area reference over disjoint pools of n draws; in parentheses the value expected from sampling noise alone)

| n draws (rays per patch) | 1 (512) | 4 (2,048) | 16 (8,192) | 64 (32,768) | 256 (131,072) | 1,024 | 4,096 |
|---|---|---|---|---|---|---|---|
| A4 binned | 0.721 (0.701) | 0.552 (0.350) | 0.511 (0.175) | 0.507 (0.088) | 0.506 (0.044) | 0.506 (0.022) | 0.506 (0.011) |
| q table | 0.685 | 0.484 | 0.420 | 0.409 | 0.408 | 0.408 | 0.408 |
| X1c | 0.738 | 0.512 | 0.368 | 0.295 | 0.275 | 0.269 | 0.269 (0.017) |
| exact direction, patch-average position | 0.603 | 0.372 | 0.224 | 0.152 | 0.120 | 0.111 | 0.111 (0.013) |
| X1 | 0.714 (1.322) | 0.473 (0.661) | 0.288 (0.331) | 0.161 (0.165) | 0.086 (0.083) | 0.048 (0.041) | 0.029 (0.021) |

The binned sender plateaus at 0.506 from about 16 draws on while its ratio stays at 1.506 for every n: the residual is bias, 46 times its
noise-only expectation at 4,096 draws. X1 follows its noise expectation all the way down (0.029 vs 0.021): no bias is detected.

### 4.4 Per-draw distribution (single K = 512 relation, 4,096 draws, vs area)

| arm | ratio: mean / sd / median / p05 / p90 / p99 / max | abs rel L1: mean / median / p90 / max | draws with ratio < 1 |
|---|---|---|---|
| A4 binned | 1.506 / 0.259 / 1.49 / 1.12 / 1.85 / 2.18 / 2.86 | 0.721 / 0.694 / 1.01 / 1.92 | 1 % |
| q table | 1.399 / 0.250 / 1.38 / 1.03 / 1.73 / 2.05 / 2.68 | 0.685 / 0.661 / 0.95 / 1.80 | 4 % |
| X1c | 1.064 / 0.519 / 0.93 / 0.65 / 1.51 / 3.41 / 6.59 | 0.738 / 0.600 / 1.10 / 5.95 | 61 % |
| exact direction, patch-average position | 1.034 / 0.323 / 0.95 / 0.70 / 1.41 / 2.29 / 3.96 | 0.603 / 0.523 / 0.90 / 3.49 | 59 % |
| X1 | 0.994 / 0.547 / 0.86 / 0.60 / 1.39 / 3.52 / 8.06 | 0.714 / 0.574 / 1.04 / 7.66 | 70 % |

X1 has a lower error than the binned sender on the same draw in 62 % of the draws. At the deployed budget the two are equally wrong
on average (0.714 vs 0.721); the exact estimator is unbiased but right-skewed (70 % of single draws under-deliver, a few over-deliver
up to 8x). The original reported median 0.62 and worst 2.2 over 64 draws; the median is confirmed (0.57), the tail is heavier than
64 draws can show (p99 3.5, max 8.1).

### 4.5 Where does the binned bias come from? (4,096 draws, paired on the same rays, vs area; SE in parentheses)

Chain in the total ratio: M64(s16) 1.506 -> M64(s_pool) 1.453 -> q 1.399 -> per-ray projection of both 1.390 -> exact direction,
patch-average position 1.034 -> X1 0.994.

| step | change of the ratio | change of abs rel L1 | share of the ratio excess (+0.512) |
|---|---|---|---|
| 16 deployed source points -> pooled source points | +0.053 (0.000) | +0.053 (0.000) | 10 % |
| incident bin projection (M64 on the pooled source -> physical q with the exact light direction) | +0.053 (0.001) | +0.045 (0.001) | 10 % |
| table lookup vs per-ray Monte Carlo of the same projection (control, should be 0) | +0.009 (0.013) | +0.009 (0.013) | 2 % |
| **outgoing direction binned -> exact** (position patch-averaged) | +0.356 (0.010) | +0.288 (0.011) | 70 % |
| outgoing direction binned -> exact (position exact) | +0.399 (0.012) | +0.383 (0.012) | - |
| position patch-averaged -> hit point (direction exact) | +0.040 (0.009) | +0.082 (0.009) | 8 % |
| position patch-averaged -> hit point (direction binned) | -0.003 (0.013) | -0.013 (0.013) | - |
| patch centre instead of patch average (direction exact): X1c - patch-average arm | +0.029 (0.008) | +0.158 (0.008) | - |

The same quantities in the WL95A definitions (my draws; the original subtracts n0 from each L1, which cancels in differences):

| WL95A quantity | original | own (vs CKF) | own (vs area) |
|---|---|---|---|
| Rq_incident = M64(s_pool) - q | 0.066 +- 0.005 | 0.045 +- 0.001 | 0.045 +- 0.001 |
| Rq_outgoing = q - X1c | 0.165 +- 0.047 | 0.125 +- 0.008 | 0.139 +- 0.008 |
| Ra = M64(s_pool) - X1c | 0.232 +- 0.047 | 0.170 +- 0.008 | 0.184 +- 0.008 |
| Rp = X1c - X1 | 0.157 +- 0.060 | 0.222 +- 0.012 | 0.240 +- 0.011 |

The original numbers are reproduced within about 1.3 of their SE, except Rq_incident (0.066 vs 0.045; the two q tables and the two pooled
sources are separate Monte Carlo estimates and differ by 1..4 % in flux; both values are "small"). The predeclared WL95A rule
(Ra >= 0.116 and above 2 SE) is still met with my numbers.

What the factorial adds: the original path attributes 0.157 (mine 0.22..0.24) to "position" through X1c -> X1, but X1c evaluates the
sender at one point per patch, the patch centre. With the direction exact, replacing the hit point by the patch **average** costs only
0.082 in L1 and +0.040 in ratio; the remaining 0.158 of the X1c error is the centre point-sampling of a function that varies inside the
patch. With the direction binned, position does not matter at all (-0.013 +- 0.013). So on this workload the outgoing-direction binning
explains 0.29..0.38 of the 0.41 map error of the ideal table and 70 % of the ratio excess; genuine position projection explains about 0.08
and 8 %.

### 4.6 Supplementary: outgoing resolution (768 further draws, seeds 9790100 + d; ideal bin average of the exact once-scattered radiance, exact incident light; `binres.json`)

| outgoing grid (rings x sectors) | exact position: ratio (SE) / abs rel L1 | patch-average position: ratio (SE) / abs rel L1 |
|---|---|---|
| 4 x 16 (the 64 bins under test) | 1.412 (0.023) / 0.441 | 1.369 (0.021) / 0.379 |
| 8 x 32 | 1.024 (0.014) / 0.120 | 1.058 (0.012) / 0.107 |
| 16 x 64 | 1.008 (0.016) / 0.044 | 1.038 (0.012) / 0.116 |
| 32 x 128 | 0.992 (0.017) / 0.036 | 1.034 (0.011) / 0.114 |
| 64 x 256 | 0.996 (0.020) / 0.031 | 1.038 (0.013) / 0.124 |
| exact direction (X1, same draws) | 0.976 (0.017) / 0.042 | - |

The 4 x 16 rows reproduce section 4.5 on a third independent set of draws. One refinement of the outgoing grid (256 cells) removes
the over-delivery (1.41 -> 1.02..1.06) and about three quarters of the map error; the patch-constant floor is about 0.11 in L1 and +3.5 % in
ratio. This is a per-ray Monte Carlo of an ideal table, not a built table: it says nothing about the cost, storage or learnability of
such a carrier, and nothing about an incident-side refinement. It does show that the bias measured under P0-2 is a property of the
4 x 16 outgoing resolution on this lobe, not of binned sender tables as a class. The worklog lists exactly this as untested
(`95A_WORKLOG...md:185`) and restricts its own statement accordingly (`:155`).

### 4.7 Supplementary: is the M64 object itself well defined? (512 further draws; `mquad.json`)

M64 averages the BRDF over each incident bin with 8 x 8 cell centres (`quad87.py:29,49-54,240-242`). At grazing incidence this could be
under-resolved. Rebuilding the four illuminated rows (incident bins 12..15) with 8 x 8, 32 x 32, 128 x 128 cell centres and with 131,072
uniform random samples per bin pair changes the delivered ratio by -0.007, +0.000, -0.012 and -0.008 relative to A4 (1.459 +- 0.010 on
these draws). The M64 object is converged in its incident quadrature to about 1 % of the delivered flux.

### 4.8 Weight convention

The original B2 already uses the corrected 4 x 16 weights W_T: its stored reference array equals mine exactly (max abs difference 0.0;
`analysis.json: wl95a_ref_array_vs_mine`). Scoring my pooled arms against the reference built with the 8 x 8 weights as coded in `an94.W`
or with uniform weights changes the binned ratio from 1.531 to 1.535 or 1.511 and X1 from 1.008 to 1.011 or 0.995 (shared set). The
delivered side of my chain never uses bin weights on the receiver (it sums cos-weighted rays directly, checked against the analytic form
factor). The bias does not depend on the weight convention; for a Lambert receiver this is expected, since its leaving radiance is the
same in every bin.

### 4.9 Is the reference `CKF_none e2_p` trustworthy?

Own area-form reference: total exitance 15.360 (relative SE 1.6e-4), own half-split 4.6e-4.

| quantity | value |
|---|---|
| CKF total / area total | 0.985; relative SE of the CKF total 0.027 (bins x halves as replicates); z = -0.55 |
| CKF abs rel L1 vs area | 0.067 (expected from the CKF noise alone: 0.061) |
| half a vs area; half b vs area | 0.961 / 0.082; 1.010 / 0.103 |
| n0 (half-split, as the original defines it) | 0.0593 (original 0.059) |
| panel-facing face: CKF vs area | 0.989 / 0.067 |
| per-patch chi2 (27 patches with signal) | 1.90; largest abs z 3.5; mean z -0.24 |
| ring means of the CKF bins (Lambert receiver: should be flat) | 1.049, 0.970, 1.018, 0.963 of the patch mean |
| own X1 flight (4,096 draws) / area total | 0.994 +- 0.009 |

Three estimators with separate code (the production path tracer, my hemisphere flight with exact evaluation, my area-form integral) agree
on the total within 1.5 % +- 2.7 % and 0.6 % +- 0.9 %. The map error of the reference against the area form (0.067) is what its noise
predicts, and the error of the mean of the halves is about 1/sqrt(2) of the single-half errors (0.082, 0.103), which is what pure noise
does and a shared bias would not. n0 = 0.059 is the right scale (the realized error is 0.067, about 13 % above n0). I find no sign that the
reference under-counts grazing light; a deficit above about 7 % of the total would have been visible at 2 sigma. Caveats: the per-patch
chi2 of 1.9 says that the bin-replicate SE underestimates the per-patch noise (heavy-tailed sampling of a sharp lobe; two patches at
z about -2.5), and all three estimators share the scene description and the ER closure, so an error of the closure itself would not show.
Using the area reference instead of CKF lowers every ratio by about 1.5 % and does not change any conclusion.

## 5. Findings

| id | category | material | claim | original vs independent | evidence | decision-level impact | proposed verdict | confidence |
|---|---|---|---|---|---|---|---|---|
| E2-F1 | confirmation | yes | P0-2, P0-3 (implementation) | deployed arrays A4 / X1c / X1 identical to 1e-6; 1.208 / 0.524, 0.719 / 0.388, 0.764 / 0.627 both | `results/audit97/E2/analysis.json: deployed`; `e2_chain.py:311-344` | the original flight, receiver and metrics are not defective for these arms | VERIFIED | high |
| E2-F2 | confirmation | yes | P0-2 | binned pooled 1.544 / 0.546 (SE 0.035, 64 draws) vs 1.529 (0.004) / 0.529 vs CKF, 1.506 (0.004) / 0.506 vs area (4,096 draws) | `summary_tables.json: pooled_4096, wl95a_arrays` | the over-delivery of about 1.5x is real, more than 100 SE from 1 | VERIFIED (tested workload) | high |
| E2-F3 | confirmation | yes | P0-2 ("not eliminated by relation sampling") | L1 plateau 0.511 (16 draws) -> 0.506 (4,096 draws), ratio constant; noise-only expectation 0.011 | `summary_tables.json: convergence_vs_area` | the residual is bias of the 64-bin projection, not sampling | VERIFIED (tested workload) | high |
| E2-F4 | confirmation | yes | P0-3 (expectation) | X1 pooled 1.022 / 0.111 (64 draws) vs 0.994 (0.009) / 0.029 (0.007) vs area (4,096 draws); follows its noise expectation | `summary_tables.json: pooled_4096, convergence_vs_area` | exact hit-point evaluation is unbiased and closes the pooled error (0.506 -> 0.029) | VERIFIED in expectation | high |
| E2-F5 | statistical_weakness | yes | P0-3 (practical reading) | single K = 512 draw: X1 mean L1 0.714 vs binned 0.721 (median 0.574 vs 0.694); better in 62 % of draws; tail max 8.1 (original worst 2.2 of 64); 16 draws needed to halve the binned plateau, about 256 draws (131k rays per patch) to reach 0.09 | `summary_tables.json: per_draw_vs_area, convergence_vs_area` | "closes the error" holds only with roughly 16x..250x the deployed relation budget; at the deployed budget there is no gain on average. The worklog says this (`:132`), the P0 wording does not | CONDITIONALLY_VALID | high |
| E2-F6 | statistical_weakness | no | P0-3 (original X1 number) | original X1 pooled L1 0.111 is the 3rd percentile of own 64-draw pools (0.174 +- 0.044) | `summary_tables.json: wl95a_arrays.X1` | the quoted 0.111 was a favourable pool; conclusion unchanged by the 4,096-draw result | VERIFIED (conclusion), number optimistic | medium |
| E2-F7 | evaluation_defect | yes | WL95A "sender-side spatial projection also counts (Rp = 0.157)" and the "hit-point" part of P0-3 | Rp original 0.157 +- 0.060, own 0.24 +- 0.011; but factorial: position (patch average -> hit point, exact direction) 0.082 +- 0.009 L1, +0.040 ratio; centre point-sampling alone 0.158 +- 0.008; position with binned direction -0.013 +- 0.013 | `summary_tables.json: decomposition_4096_draws`; original confound disclosed at `95A_WORKLOG...md:152`, counted anyway at `:171` | the position share is about one third of what X1c -> X1 suggests; an exact-direction sender that is constant per patch already gives 1.034 / 0.111. Sub-patch position is a second-order term on this workload | CONDITIONALLY_VALID (Rp as an attribution: INVALID_COMPARISON for "position", valid as "centre vs hit point") | high |
| E2-F8 | confirmation | yes | P0-2 (incident vs outgoing) | original Rq_incident 0.066 +- 0.005, Rq_outgoing 0.165 +- 0.047; own: source points +0.053, incident bin +0.053 (L1 0.045), outgoing direction +0.356 ratio / 0.288..0.383 L1 | section 4.5 | about 70 % of the over-delivery is outgoing-direction binning, about 20 % incident side (half of it the 16 fixed source points), about 8 % position | VERIFIED | high |
| E2-F9 | dataset_limitation | yes | P0-2 (generality over bin layouts) | untested in the original (`:185`); own: 4 x 16 -> 8 x 32 outgoing cells: ratio 1.41 -> 1.02..1.06, L1 0.44 -> 0.11..0.12; 16 x 64: 1.008 / 0.044 | `binres.json` | the bias is specific to the 4 x 16 outgoing resolution; it is not evidence against binned or tabular sender carriers as a class. No cost or learnability statement follows | CONDITIONALLY_VALID (P0-2 must stay scoped to 64 bins) | medium-high (ideal per-ray table, one workload) |
| E2-F10 | confirmation | yes | reference validity | CKF vs own area form: 0.985 +- 0.027, L1 0.067 (noise expectation 0.061), n0 0.0593; X1 vs area 0.994 +- 0.009 | section 4.9 | reference trustworthy at its stated noise; no grazing under-count above about 7 % | VERIFIED | high (shared closure and scene are not independently tested) |
| E2-F11 | statistical_weakness | no | reference noise model | per-patch chi2 1.9 against the bin-replicate SE; max abs z 3.5 | `summary_tables.json: reference_vs_area` | per-patch reference errors are somewhat larger than a Gaussian n0 reading suggests; totals unaffected | NONE | medium |
| E2-F12 | confirmation | no | weight convention | W_T vs 8 x 8 as coded vs uniform: binned 1.531 / 1.535 / 1.511; original B2 reference = W_T exactly | section 4.8 | no dependence on the weight convention on the Lambert box A | VERIFIED | high |
| E2-F13 | confirmation | no | M64 object | 8 x 8 midpoint incident quadrature vs 32 x 32, 128 x 128, 131k random: delivered ratio within 0.012 | `mquad.json` | the object under test is converged; the bias is not a build artifact of M64 | VERIFIED | high |
| E2-F14 | provenance_gap | no | reproducibility | hashes of `deliv.npz`, `ops95a.npz` equal between server and local; `A4` regenerates to 1.7e-6; B2 table reproduces from `deliv.npz` with own metric code (1.544 / 0.546, 1.422 / 0.433, 1.037 / 0.268, 1.022 / 0.111). No gap found | `analysis.json: wl95a_pooled_recomputed`, `senders.json` | none | VERIFIED | high |

## 6. P0 assessment within this lens

**P0-2: VERIFIED on the tested workload** (CK2 `none`, one point light at 81..89 degrees incidence on the sharp panel, one static geometry,
Lambert receiver box A, exactly-2-event panel-mediated light, 4 x 16 bins). The pooled binned over-delivery is real (1.506 +- 0.004 against
an independent reference, 1.529 against the original one; original 1.544 +- 0.035), it does not shrink with relation sampling (map error
0.506 from 16 to 4,096 draws), and it does not depend on the weight convention or on the M64 build quadrature. Restrictions that belong to
the claim: (i) magnitude and sign (over-delivery) are those of this receiver in this geometry; a bin average moves lobe energy between
directions, so other receivers can be under-served, which I did not measure; (ii) about 70 % of it is outgoing-direction binning, 20 %
incident side, under 10 % position; (iii) it is a 64-bin statement: one refinement of the outgoing grid removes the ratio bias and three
quarters of the map error (E2-F9). Read as a statement about binned sender carriers in general it would be an UNTESTED_INFERENCE, and my
supplementary measurement speaks against that reading.

**P0-3: CONDITIONALLY_VALID.** In expectation the claim is verified and even stronger than reported: exact direction plus hit point is
unbiased (0.994 +- 0.009) and its pooled map error falls to 0.029 against an independent reference. The conditions: (i) the gain exists
only with many relation draws; at the deployed 512 rays per patch the exact estimator is as wrong as the binned one on average (0.714 vs
0.721) and worse on the deployed draw (0.634 vs 0.507), with a heavy right tail; (ii) "exact direction" carries almost all of it: with a
patch-constant position the result is already 1.034 / 0.111, and the original position share is inflated by the patch-centre arm (E2-F7);
(iii) "continuous" is not shown to be necessary: 8 x 32 or 16 x 64 outgoing cells give 0.12 or 0.044 (E2-F9); (iv) one light, one geometry,
Lambert receiver, first flight only. Nothing here measures the image-level (ROI) effect, the learned sender, or later bounces.

Neither verdict supports rejecting or accepting an architecture class beyond "a patch-constant 4 x 16 outgoing table cannot carry this
grazing sharp lobe to this receiver without bias, and an unbiased per-ray evaluation needs far more than 512 rays per patch".

## 7. Special checks

| check | result |
|---|---|
| ER cosine convention against an analytic Lambert case | pass (second argument; 1.8e-7) |
| own flight and own area estimator against Lambert's polygon form factor | pass (0.99986 and 0.99989) |
| correct 4 x 16 weights | original B2 uses them; no effect on box A (E2-F12) |
| reference noise scale n0 | 0.0593 reproduced; realized error against the area form 0.067 |
| reference grazing under-count | not found (0.985 +- 0.027) |
| deployed realization identity | relation offsets bitwise; arrays to 1e-6 |
| independent seeds, >= 64 pooled draws, per-draw distribution, SE | 4,096 + 768 + 512 draws; sections 4.2..4.4 |
| receiver mask | box A = 1130..1183 = PSID 7, material 4 asserted in `e2_chain.py:49`; panel face = first 9 |
| metric compatibility | all arms and both references are per-patch exitance with channels summed; ratio and L1 as the original defines them (its B2 table recomputes exactly) |

## 8. Corrections to my own working notes (first attempt) and incidents

- The analytic form-factor routine had an operator-precedence bug that crashed before producing any number; fixed before the first result.
- The first attempt built an "own M64" with 4,096 samples per bin pair; it delivered 0.075 less than A4. That estimate was
  under-sampled for a sharp lobe. The properly sampled rebuild (section 4.7) agrees with A4 within 0.012. The `M64own_spool` arm in
  `analysis.json` and in the flight files should be ignored.
- `e2_analyze.py` printed NaN for the per-patch chi2 (patches with zero signal); `e2_tables.py` restricts it to the 27 patches with signal.
- The first version of the M64 quadrature job launched one kernel per incident cell and ran about 9 min at full GPU load without
  finishing. I stopped it and rewrote it to a few launches (123 s). When stopping it I matched my own remote shell with `pgrep -f` as
  well; only my own two processes were affected.
- `summary_tables.json` and `analysis.json` were regenerated in this session from the stored flight files; the printed tables of the first
  attempt were reproduced unchanged.

## 9. Not verified

- The learned sender arms (A0) and the rank arms (A1..A3) of WL95A, section B1 (sender field metrics, passivity) and B3 (ROI renders).
- Any other light, geometry, material, receiver type (non-Lambert), the panel-face-only tables beyond the totals quoted, and bounces
  beyond the first flight.
- Under-delivery elsewhere in the scene that must accompany the over-delivery to box A.
- The cause of the 4 extra rays in my tracing of the deployed relation (no effect on any delivered value).
- The ER closure and scene description themselves (shared by all three estimators).
- Whether a built 256-bin table (with its own incident projection, storage and sampling) behaves like the ideal per-ray projection of
  section 4.6.
- The original's SE formulas line by line; I compared its SE values with my bootstrap only (0.035 vs 0.031..0.034 for the binned ratio).

## 10. Reproducibility appendix

Inputs (sha256 prefix): `~/rest_x94/results/fields/CKF_none.npz` 3a6238a8a42d1db4 (2 x 4,096 spp, seeds 94301 / 94302);
`~/rest_x95a/results/deliv.npz` f2324a32effc80ca (= local `results/ecro_attr95a/deliv.npz`); `~/rest_x95a/results/ops95a.npz`
d4d02e5a2f423ce9; `~/rest_x95a/results/senders.npz` 5d365d19d6dffadc; frozen code `~/rest_x97/E2/code` = commit c152120.

Own scripts (`experiments/audit97/E2/`, sha256 prefix): `e2_chain.py` adb3ff7f96327a4f, `e2_extra.py` e1a6ba730c71e3dd,
`e2_analyze.py` 0c253ec3de45cd4a, `e2_tables.py` c55187711d37a41d.

Own outputs (`results/audit97/E2/`; server copies in `~/rest_x97/E2/results/`): `checks.json` 2d01c7c94a3c515a, `senders.json`
8843132bcd5512fe, `senders_e2.npz` c6a02b0b8dc61efa, `flight_deployed.npz` 7111755da95da183, `flight_shared2048.npz` 7da85358039e2b8a,
`flight_indep2048.npz` a5f18db50d9b7cc3, `aref_main.npz` bab8ea5149fffb56, `ref_boxA.npz` 14c1f69c961482c5, `wl95a_deliv_boxA.npz`
92e22a16504959fc, `binres.json` 57114da6799cf21c (`binres.npz` on the server only, 7eba902c8207eb3e), `mquad.json` 44821526e8cbe9a6,
`analysis.json`, `summary_tables.json`, printed tables `analysis.txt`, `summary_tables.txt`, and the action log of the interrupted
first attempt `_interrupted_run_log.md`.

Commands (server, after `. ~/rest_x95a/env.sh`, in `~/rest_x97/E2`, `R=~/rest_x97/E2`):

```
$PY e2_chain.py checks  $R                               # rng 9700001, 9700100+d, 9700200, 9700300+d, 9700400
$PY e2_chain.py senders $R                               # rng 9710001..9710005; deployed offsets default_rng(86001)
$PY e2_chain.py deployed $R                              # relation seed 86001; projection rng 9730001
$PY e2_chain.py flight  $R 2048 shared2048 shared 9720000   # draw d: default_rng(9720000 + d), offsets shared by the 54 patches
$PY e2_chain.py flight  $R 2048 indep2048  indep  9740000   # draw d: default_rng(9740000 + d), independent per patch
$PY e2_chain.py aref    $R 24 16 9750000 main            # 16 batches x 2^24 samples per patch, PCG32
$PY e2_extra.py binres  $R                               # 768 draws, seeds 9790100 + d, projection rng 9790001
$PY e2_extra.py mquad   $R                               # 512 draws, seeds 9780000 + d, build rng 9770001..9770005
```

Local (numpy only): `python experiments/audit97/E2/e2_analyze.py results/audit97/E2` and
`python experiments/audit97/E2/e2_tables.py results/audit97/E2` (bootstrap rng 9760001 and 9761001).

Run times: checks 12 s, senders 22 s, flights about 1 min each, area reference 74 s, binres 32 s, mquad about 3 min; GPU peak 2.7 GB.
