# WL97 Phase 2 - Track J1: adversarial adjudication of cluster P0-1 (learned S1 sender: "large avoidable error"; passivity)

Date 2026-11-16. Independent reviewer report. Audit only: nothing trained, no existing file modified, no commit.
Own files: `experiments/audit97/J1/`, `results/audit97/J1/`, server `~/rest_x97/J1/`.

## 0. Summary

- **"Large error" stands.** Three phase-1 tracks (E1, C2, A; B digit-exact on the albedo) reproduce the WL95A learned-sender numbers, and the stored
  `results/ecro_attr95a/{main,an}.json` values I re-read agree with every cited figure.
- **"Avoidable" is not established as worded.** WL95A itself wrote "Not shown" (line 143). What "avoidable" is measured against decides the answer
  (section 3). Nobody determined the best error a patch x 64-bin operator can reach against the path-traced reference.
- **The rank-16 conflict is resolved (decisive check, section 2).** On CK2 the rank-16 "inadequacy" is **allocation, not capacity**: an explicit
  non-negative rank-16 material-level operator is exact on the CK2 incident support and costs 0.42 -> 0.54 training-weighted row L1 (it is as good as
  the WL95A A1 bound arm elsewhere). B-06's "structural, not a fit-distribution artefact" is refuted for CK2. What *is* structural: no light-independent
  rank-16 operator represents S1's M64 over all incidences (Eckart-Young 0.46-0.52 flux-Frobenius; rank 46-59 for 10 %), so allocation is zero-sum.
- **WL95A's RANK16_DELIVERY rule measured distance to M64's delivery**, and M64 is itself 1.54x the path-traced reference. The label says nothing
  about the physical target and nothing about capacity. WL96 (lines 57-60) already reconciled this; the roadmap's WL95A entry still carries the bare label.
- **Passivity is correctly derived and now also measured on the WL86 deployed checkpoints** (new: max directional albedo 2.70 / 2.21 on S1, 15.9 % /
  13.3 % of rows above 1). It is a worst-row property of the local factor pair. On the CK2 input all checkpoints emit less than they receive in
  total. No worklog draws a causal or stability consequence from it; none would be justified (no recursion in `model86.apply`).

## 1. Method

For each finding I re-read the cited code and stored numbers, looked for the usual over-eager-auditor mistakes, and ran two checks of my own where
phase-1 tracks disagreed or left a gap:

| check | script | where | output |
|---|---|---|---|
| J1-a exact attainable frontier of signed rank-16/32/48 operators (row-weighted Eckart-Young; lower bound for any non-negative operator) | `experiments/audit97/J1/j1_rank16_pareto.py` | local numpy | `results/audit97/J1/j1_rank16_pareto.json` |
| J1-b explicit **non-negative** rank-16 operator exact on the CK2 support + NMF12 elsewhere | `experiments/audit97/J1/j1_rank16_nonneg_construct.py` | local numpy | `results/audit97/J1/j1_rank16_nonneg_construct.json` |
| J1-c directional albedo of the local factors of the WL86 deployed, WL92A OWN and WL92A SUM checkpoints (CK2 geometry, a95 convention) | `experiments/audit97/J1/j1_wl86_passivity.py` | LabServer63, one GPU job < 1 min, read-only inputs | `results/audit97/J1/j1_wl86_passivity.json`, `j1_wl86_passivity_server.log` |

Not redone: the sender-field and delivery reproductions (confirmed by E1, C2 and A independently), the training-distribution stage of E1, the
ten-checkpoint dump of C1. Findings resting only on those are marked "not re-run" below.

Inputs: `results/ecro_attr95a/ops95a.npz` (A4 = M64 S1, A1-A3 bound arms), incident shares `p_tr` (WL92A training panel irradiance per incident
bin) and `p_ck` (CK2) from `results/audit97/B/learned_s1_albedo.json`.

## 2. Decisive check: is rank 16 a capacity limit on CK2?

### 2.1 Operator families

| family | definition | relation to the deployed model |
|---|---|---|
| F-mat | one non-negative rank-16 operator per material, 64 x 64 bins x rgb, shared by all S1 patches and all lightings | the WL95A bound arms A1/A2 (and WL92's Eckart-Young bound) |
| F-patch | per-patch `a_n c_n^T` (softplus factors, R = 16) generated from geometry tokens, light-independent | **the deployed ECRO local operator** (`experiments/ecro86/model86.py:100-107`); contains F-mat |
| F-light | an operator fitted to one lighting | not a reusable operator; rank 1 per patch suffices |

### 2.2 Results

All errors are relative to M64 (A4), i.e. to the **binned** material operator, not to the physical response.

**All-incidence capacity (J1-a spectrum).** Best rank-k relative flux-Frobenius error of M64 S1, R / G / B:
rank 16: 0.520 / 0.458 / 0.461 (equals WL92's stored 0.52 / 0.46 / 0.46); rank 32: 0.32 / 0.23 / 0.24; rank 48: 0.18 / 0.07 / 0.09;
rank needed for 10 %: 59 / 46 / 47. The four CK2 incident rows (bins 12-15) have rank 4.

**Exact frontier, signed rank 16 (J1-a; a lower bound for every non-negative rank-16 operator, whatever the fit or training):**

| weight on CK2 objective | CK2-weighted rel L2 | training-weighted rel L2 (flux form) | same, per unit irradiance |
|---|---|---|---|
| 0 (training optimum) | 0.959 | 0.331 | 0.462 (CK2 0.790) |
| 0.1 | 0.192 | 0.383 | 0.486 (CK2 0.185) |
| 0.5 | 0.039 | 0.404 | 0.498 (CK2 0.054) |
| -> 1 (CK2 exact) | 0.000 | 0.407 | 0.504 |

Rigorous bounds from the supporting lines: any rank-16 operator with CK2 error <= 0.05 has training-weighted error >= 0.43 / 0.40 / 0.40 (R / G / B, flux
form); conversely training error <= 0.3 forces CK2 error > 1. At rank 32 the same trade is 0.134 -> 0.158; at rank 48 0.035 -> 0.043.

**Explicit admissible operator (J1-b; non-negative, rank 16 per colour, max albedo 0.975):**

| operator | CK2 field rel L1 / energy / peak | training-weighted row L1 (non-CK2 rows) | training rel L2 | op flux L1 | ring 0-3 op L1 |
|---|---|---|---|---|---|
| NMF16, training weights (reproduces B's `train` fit) | 0.891 / 0.348 / 0.014 | 0.422 (0.411) | 0.506 | 0.571 | 0.96 / 0.89 / 0.65 / 0.34 |
| **CK2 rows exact + NMF12 elsewhere** | **0.000 / 1.000 / 1.000** | 0.544 (0.555) | 0.561 | 0.669 | 0.74 / 0.96 / 0.84 / 0.43 |
| ring 0 exact (the F0/C2/WL96 counterexample) | 0.000 / 1.000 / 1.000 | 0.953 | 0.965 | 0.936 | 0 / 1 / 1 / 1 |
| NMF12 only (reference) | 0.898 / 0.346 / 0.011 | 0.560 | 0.585 | 0.680 | 0.98 / 0.96 / 0.84 / 0.42 |
| WL95A A1 (NMF16, flux-Frobenius) | 0.921 / 0.458 / 0.028 | 0.573 | 0.552 | 0.603 | 0.97 / 0.86 / 0.68 / 0.40 |
| WL95A A2 (L1-NMF16) | 0.892 / 0.111 / 0.002 | 0.351 | 0.612 | 0.380 | 0.89 / 0.74 / 0.32 / 0.22 |
| WL95A A3 (rank 32) | 0.363 / 0.669 / 0.076 | 0.134 | 0.368 | 0.233 | 0.52 / 0.38 / 0.16 / 0.19 |

### 2.3 Which sentence is true for which family

1. **F-mat, all incidences: capacity-limited (true).** No light-independent rank-16 operator in the 64-bin basis represents S1's M64 over the incidence
   range (>= 0.46 flux-Frobenius; the best admissible fits are 0.38-0.60 flux L1). WL92 measured this and labelled representation INCONCLUSIVE; that stands.
2. **F-mat on CK2: not capacity-limited (true).** An admissible rank-16 operator is exact on the CK2 incident support (rank 4 needed). Buying it costs
   four components: training-weighted row L1 0.42 -> 0.54, i.e. rank-12 quality elsewhere. That operator is **not worse than WL95A's own A1 bound** on the
   training-weighted row L1 (0.544 vs 0.573) and close in L2 (0.561 vs 0.552) and op flux L1 (0.669 vs 0.603). The "ring-0 exact" counterexample cited by
   F0-08 / C2-05 / WL96 is correct but degenerate (0.95 elsewhere); the CK2-support version is the non-trivial one.
3. **B-06's reading is refuted for CK2.** Its numbers reproduce (my NMF16 0.891 / 0.348 / 0.014, 0.422 vs B's 0.899 / 0.353 / 0.015, 0.423). Its
   inference fails for two reasons: (a) its incidence weights multiply rows that already carry z_m, so a "12 % CK2 weight share" is a far smaller share
   of the objective (ring-0 rows have z = 0.125); (b) multiplicative-update NMF is a local optimiser and never visits the dedicated-component solution.
   B's own table shows the mild trade (CK2 L1 0.90 -> 0.59 for training L1 0.423 -> 0.450).
4. **F-patch (deployed): no capacity reading is supported.** F-patch contains F-mat, so statement 2 holds a fortiori; for the single CK2 lighting
   each patch needs rank 1. The learned senders' CK2 field error (rel L1 1.04 / 1.12 vs M64) is therefore far above what the class admits on that
   input (0), and on all incidences the learned operators (op flux L1 0.96-1.04) are far above the admissible rank-16 fits (0.38-0.60). The gap is
   identification / optimisation / generation, not rank. This is existence, not learnability: WL94D's direct sender supervision moved the
   training-field error only 0.82 -> 0.72 in a 1,000-step smoke.
5. **F-light:** exact at rank 1 per patch; irrelevant to a reusable operator.

### 2.4 What WL95A's RANK16_DELIVERY rule tested

The frozen rule (`experiments/ecro_attr95a/protocol/ecro_attr95a.json` /decisions/RANK16_DELIVERY; code `a95.py:494-512`) asks whether A1 or A2 is passive,
delivers box-A flux within 10 % of **A4 = M64**, has a box-A map within 0.10 rel L1 of **A4**, and an ROI E_H no worse than A4's plus tolerance.
Stored outcome (`an.json` rank16): A1 flux 0.631, map 0.649; A2 0.158, 0.842; label INADEQUATE_ON_TESTED_WORKLOAD follows the rule.

Against the path-traced reference (`an.json` delivery, pooled, box A): A4 1.544x / abs rel L1 0.546; A1 0.975x / 0.655; A2 0.244x / 0.762;
A3 1.127x / 0.386; learned s8601 0.517x / 0.569; s8602 1.630x / 0.965; exact per-ray X1 1.022x / 0.111. ROI E_H: floor 0.0118, A1 0.0209,
A3 0.0204, A4 0.0245, s8601 0.0222, s8602 0.0564.

So: (i) the comparator is biased; a rank-16 operator equal to M64 would "pass" and deliver 1.54x; (ii) A3 (rank 32) "fails" the rule while being
closer to the reference than M64 (0.386 vs 0.546); (iii) the bound arms were fitted to all incidences (flux-Frobenius / W94 weights; ring-0 rows carry
6.4 % of the objective), so their failure at grazing is the allocation of 2.2. **"INADEQUATE" is a valid rule outcome meaning "all-incidence
rank-16 fits do not reproduce M64's delivery at CK2". It carries no statement about the physical target and none about rank-16 capacity.**
WL95A discloses most of this itself (line 122: A1 0.98 vs A4 1.54 against the reference; line 144: "[INF] capacity / fit-distribution limit ... a
grazing-weighted rank-16 fit was not tried"; line 166: the ROI criterion passes only because M64 over-delivers).

### 2.5 The statements that travelled

- **"Rank-16 loses the panel lobe"** (WL92 roadmap entry, WL92:138/222). Supported in the stated norm and scope: material-level, all incidences,
  Eckart-Young 0.46-0.52 flux-Frobenius (reproduced exactly), NMF16 keeps 47-54 % of mirror-bin flux. "Loses" means "keeps about half" over all
  incidences and <= 0.03 at CK2 grazing for all-incidence fits. It is not a statement about CK2-aware or geometry-conditioned operators.
- **"Learned S1 operator at the zero-operator level"** (WL95:244/313/444/491/599, roadmap WL95 entry). Supported in the flux-L1 norm only
  (0.96-1.04 vs zero 1.000). In flux-Frobenius the same operators are 0.77-0.83 (C1, OWN-family checkpoints) and 0.88-0.89 (WL92:138, WL86
  checkpoints) vs zero 1.0 and Eckart-Young 0.46-0.52, with mass 0.68-1.06 and peak retention 0.11-0.28. Flux L1 saturates near 1 for any
  operator with the mass in the wrong bins (the Frobenius-optimal rank-16 operator scores 0.891). The conclusion drawn from it at WL95:599 ("does not
  support ... as trained") survives; the descriptor overstates. The teacher M64 is itself biased at delivery.
- **WL94D's "0.46-0.52 floor"**: WL94D:147 already says "a different norm"; WL95 section 2.2 corrects the comparison. The qualitative conclusion
  (learned is well above the admissible rank-16 level) holds in matched norms (L1 1.0 vs 0.38-0.60; Frobenius 0.8 vs 0.5).
- **WL96:57-60** ("fit-distribution allocation and light-independence, not rank capacity; moot for design"). Correct for CK2. Two qualifications:
  allocation is zero-sum because of the all-incidence capacity limit (cost quantified in 2.2), and the physics critic's "0.062 on CK2 but 0.85 / 0.69 on
  two other lights" is a pure-CK2-weighted fit, not the best compromise.

## 3. "Avoidable" relative to what?

| comparator | is the learned sender's CK2 error avoidable against it? | status |
|---|---|---|
| M64(s16) at sender-field level, inside the deployed class | Representationally yes: an admissible rank-16 operator has zero CK2 field error vs M64 (2.2). Whether training reaches it is not shown; WL94D's smoke says not at that budget | existence proven (J1-b); learnability untested |
| path-traced reference at box-A delivery, inside the patch x 64-bin carrier, using M64 as the class optimum | s8601: no (0.569 vs 0.546; paired difference +0.023 +/- 0.035, C2). s8602: yes (0.965 vs 0.546). ROI E_H: s8601 0.0222 vs A4 0.0245 (inside tolerance), s8602 0.0564 | seed-dependent |
| path-traced reference, best patch x 64-bin operator | Unknown. M64 is not the delivery-optimal binned operator: A3 (rank 32) reaches 0.386 and A1 has the right total flux (0.975x). The class optimum was never computed | **not measured** |
| continuous exact-direction / hit-point ER (X1, 0.111 pooled) | Yes, by 0.46-0.85 abs rel L1; but X1 is not a learned operator of this class and needs 64 pooled relation draws | outside the class (P0-2 / P0-3 territory) |

Hence the error budget "a learned operator of this class could reach" on CK2 delivery is bounded only loosely: it is at most the binned-carrier level
(about 0.4-0.55 abs rel L1 for the operators tried) and its true floor is unknown. "Large avoidable error" compresses three different statements and
only the first row is established, as an existence result.

## 4. Passivity

- **Derivation: correct.** `alb[n, m, c] = sum_d (sum_r c[n,m,c,r] a[n,d,c,r]) W_T[d]` with `W_T = cos x dOmega` exact for 4 x 16 equal-solid-angle bins
  (`a95.py:27,147-152`). The source encoding is irradiance per incident bin, confirmed by the Lambert control: learned Lambert-0.7 patches give
  irradiance-weighted exitance / irradiance 0.67-0.70 and mean albedo 0.71-0.76 (J1-c; also E1, A). M64 S1 max is 0.825 in the same convention.
  For a non-negative operator the worst-case input is a single incident bin, so "max row albedo > 1" is exactly "not passive".
- **Checkpoints (J1-c, CK2 geometry, S1 = 42 patches x 64 bins x 3 channels):**

| checkpoint (sha256 prefix) | max albedo | mean | rows > 1 | patches with any row > 1 | CK2-lit rows > 1 (max) | CK2 once-scattered exitance / irradiance on S1 | S1 patches with out > in |
|---|---|---|---|---|---|---|---|
| WL86 deployed ecro_s8601 (9d332cd6) | **2.698** | 0.558 | 15.9 % | 42 / 42 | 2.1 % (1.12) | 0.097 | 0 |
| WL86 deployed ecro_s8602 (4d993121) | **2.212** | 0.528 | 13.3 % | 35 / 42 | 1.6 % (1.08) | 0.148 | 0 |
| WL92A OWN s8601 (739e7eb5) | 2.2637 | 0.561 | 10.4 % | 42 / 42 | 4.2 % (1.44) | 0.283 | 1 |
| WL92A OWN s8602 (cee9a09a) | 3.0774 | 0.552 | 15.4 % | 42 / 42 | 41.3 % (1.79) | 0.582 | 19 |
| WL92A SUM s8601 (3ec1d591) | 1.486 | 0.279 | 0.35 % | 9 / 42 | 0 % (0.55) | 0.206 | 0 |
| WL92A SUM s8602 (d911c6b2) | 1.906 | 0.484 | 8.8 % | 41 / 42 | 0 % (0.98) | 0.189 | 0 |

  The OWN rows reproduce WL95A (2.2637 / 3.0774), B, A and E1 to all printed digits. The WL86 and SUM rows are new. C1 adds eight WL94-series
  checkpoints (not re-run). Physical q on the same input: exitance / irradiance 0.67 (bin measure, E1).
- **Scope.** The statement is about the local factor pair. `model86.apply` (`model86.py:109-124`) applies it twice in series plus a low-rank global term and never
  iterates, so albedo > 1 cannot cause divergence of the deployed model. The WL86 protocol says passivity is not imposed
  (`ecro86.json:12`, WL86:41). A passivity criterion for the composed multi-event operator was neither defined nor tested anywhere.
- **Consequences drawn in the worklogs.** WL95A:57-58/119/143, the roadmap WL95A entry, WL96:52 and WL96:136 state the fact descriptively; WL96:136
  uses it as a design caution ("non-negativity does not guarantee passivity of the resolvent"). I found no worklog sentence that attributes the CK2
  delivery error or an instability to it. E1-F5 / A-06 are right that such an attribution would be untested: on the CK2 input every checkpoint is
  sub-unity in total, and s8602's 1.63x box-A over-delivery coexists with 0.87 of the physical sender energy (directional misallocation; an inference,
  no counterfactual run).

## 5. Rulings

| finding | ruling | note |
|---|---|---|
| E1-F1 | UPHELD | Stored WL95A B1 rows match; independently reproduced by C2 (not re-run by me) |
| E1-F3 | UPHELD | Consistent with C2's redraw SDs (0.036 / 0.108); not re-run |
| E1-F4 | UPHELD | Maxima and Lambert control reproduced by J1-c; extended to WL86 deployed and SUM |
| E1-F5 | UPHELD | Energy numbers 0.283 / 0.582 and 1 / 19 patches reproduced; non-recursion verified in code |
| E1-F6 | UPHELD_WITH_MODIFICATION | Its signed-SVD "indication" is now an exact frontier plus an explicit non-negative operator; the cost of the allocation is 0.42 -> 0.54 training row L1 |
| E1-F7 | UPHELD | `an.json` pooled rows verified; add ROI E_H s8601 0.0222 vs A4 0.0245 |
| E1-F8 | UPHELD | Agrees with WL94D's stored 0.82 on training fields; E1's training stage not re-run |
| E1-F10 | UPHELD_WITH_MODIFICATION | "Two checkpoints" applies to the CK2 sender / delivery error. The non-passivity is now measured on 14 checkpoints (2 WL86, 2 SUM, 10 in C1 incl. OWN) |
| A-06 | UPHELD | All quoted maxima and distribution figures reproduce |
| A-07 | UPHELD_WITH_MODIFICATION | Correct, but WL95A already says "Not shown" (143) and discloses the reference-side numbers (122, 166). The defect is the proposition / P0-1 wording and the bare label in the roadmap entry, not concealed evidence |
| B-05 | UPHELD_WITH_MODIFICATION | Digit-exact. "Bins with albedo > 1 carry 52-62 % of training irradiance" counts a bin if any patch / state exceeds 1; the entry-level share is 10-12 % (E1-F4) |
| B-06 | **REFUTED** (headline and decision impact) | Numbers reproduce; the inference "structural, not a fit-distribution artefact ... rejection of that mechanism for sharp grazing lobes is justified" fails (2.3 item 3). Surviving part: the all-incidence capacity limit and a non-zero trade-off |
| C1-05 | UPHELD | WL94D:109/147 and WL95:230/244/599 verified; WL94D:147 itself flags "a different norm"; WL92:138 gives an independent Frobenius figure (0.88-0.89) for the WL86 checkpoints |
| C1-09 | UPHELD_WITH_MODIFICATION | Non-passivity upheld and extended. The "bin 12 patch-mean albedo 1.24-1.68" detail was not verified and is the least-lit CK2 bin (2.3 % of CK2 panel irradiance); the CK2 irradiance-weighted albedo is 0.10-0.58 |
| C2-05 | UPHELD | Paired comparisons not re-run; stored pooled values verified. J1-b supplies the two-sided version of its "one-sided" counterexample |
| C2-06 | UPHELD | Reproduced where overlapping with J1-c |
| F0-08 | UPHELD_WITH_MODIFICATION | "Allocation, not capacity" is right for CK2; "no measurement establishes a rank-16 capacity limit" is too strong - WL92's Eckart-Young result is a valid all-incidence capacity limit (reproduced), which is why allocation costs something. WL94D disclosed the norm mismatch itself; cite WL93D:229 checks out in the main WL93D file |
| F0-12 | UPHELD_WITH_MODIFICATION | Correct at phase-1 time. The gap is closed by J1-c: the WL86 deployed checkpoints' local factors are non-passive too. Still unmeasured: any passivity property of the composed operator |

## 6. P0-1 verdicts

| claim half | verdict | holds under these restrictions |
|---|---|---|
| "The learned S1 sender has large error" | **VERIFIED** | Two WL92A OWN checkpoints; CK2 "none", one grazing light, rigid static geometry. Field: rel L1 1.04 / 1.12 vs M64(s16), 1.08 / 1.19 vs physical q; peak 0.03-0.13. Delivery to box A vs path-traced reference: 0.52x / 1.63x, abs rel L1 0.57 / 0.97; seeds err in opposite directions. In-distribution: about 0.8 rel L1 vs M64 at roughly correct energy (E1, WL94D) |
| "... avoidable" | **UNTESTED_INFERENCE** | True as an existence statement inside the class at field level vs M64 (J1-b). Not shown by learning. At delivery vs the reference inside the binned carrier it is seed-dependent (s8601 no, s8602 yes), and the class optimum against the reference was never computed. True against continuous exact ER, which is outside the class |
| "The deployed learned operator may violate passivity" | **VERIFIED** for the local factor pair (WL86 deployed, WL92A OWN and SUM, WL94-series); **CONDITIONALLY_VALID** as a statement about "the deployed operator" | Worst-row property (10-16 % of S1 rows for the WL86 / OWN checkpoints); total once-scattered energy on the tested input is sub-unity; no recursion, so no stability consequence; no causal link to the CK2 delivery error is tested; the composed operator has no tested passivity property |
| WL95A RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD (P0-1 / P0-5 link) | **CONDITIONALLY_VALID** as a rule outcome; **INVALID_COMPARISON** if read as physical-target inadequacy or as rank-16 capacity | Means: all-incidence rank-16 fits do not reproduce M64's delivery at CK2. Comparator M64 is 1.54x the reference |

## 7. Affected worklog statements

| # | original claim (location) | defect or limit | corrected interpretation | known dependents | action |
|---|---|---|---|---|---|
| 1 | "The deployed learned S1 sender is a major avoidable error source" (WL95A:6, proposition; P0-1 wording) | "Avoidable" not shown (WL95A:143 says so); comparator-dependent | Large error source on CK2 for two OWN checkpoints; avoidability inside the tested carrier not established; class optimum against the reference unknown | WL95 section 2.1 basis, WL96 exclusion of learned sharp carriers | QUALIFY |
| 2 | "On CK2 it is a large error source in delivery: 0.52x / 1.63x ... seed-signed ... non-passive" (WL95A:143) | none in the numbers; s8601 is at the binned-carrier level (0.569 vs M64 0.546) | Keep; add that s8601 is not clearly worse than the exact binned sender at delivery or ROI level | WL95A NEXT_DESIGN_TARGET (UNDETERMINED) is consistent | PRESERVE |
| 3 | "[INF] This is a capacity / fit-distribution limit for grazing S1 rows, consistent with WL95's empirical ceiling" (WL95A:144) | The "capacity" alternative is false for CK2: an admissible rank-16 operator is exact there | Fit-distribution allocation of a rank budget that is too small for all incidences; CK2 exactness costs 0.42 -> 0.54 training row L1 | WL96:57-60 (already corrected), roadmap WL95A entry | QUALIFY |
| 4 | RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD (WL95A:159-166; roadmap WL95A entry) | Comparator A4 = M64 is 1.54x the reference; arms fitted to all incidences | Rule outcome only: "all-incidence rank-16 fits do not reproduce M64's delivery at CK2"; A1 is 0.975x and A3 0.386 abs rel L1 against the reference | WL96 reconciliation; any reading of rank 16 as the cause of the sharp-delivery error | QUALIFY (label preserved as history) |
| 5 | "RANK16 INADEQUATE = fit allocation (exact rank-16 on ring-0 support; moot because an exact M64 copy keeps the binned bias)" (WL96:57-60; roadmap WL96 entry) | Counterexample degenerate (0.95 elsewhere); "not rank capacity" true for CK2 only | Keep; the non-degenerate version is the CK2-support operator; the all-incidence capacity limit remains and makes allocation zero-sum | WL96 NO_VIABLE_LEARNED_MECHANISM ("non-compact for light-independent linear forms") is supported by the all-incidence result | PRESERVE |
| 6 | "A material-level rank-16 bin factorisation loses the sharp panel's mirror lobe (Eckart-Young 0.46-0.52)" (WL92:138/222; roadmap WL92 entry) | none; reproduced (0.520 / 0.458 / 0.461) | Material-level, all incidences, flux-Frobenius; "loses" = keeps about half overall | WL93D:229, WL94D:147/156, WL95 section 2 | PRESERVE |
| 7 | "cf. WL92: the rank-16 class ceiling plus learning beyond it" (WL93D:229) | Ceiling quoted across norms | Matched norms: Frobenius about 0.8 learned vs 0.5 bound; L1 about 1.0 vs 0.38-0.60 | WL93D H_R "supported" unaffected in direction | QUALIFY |
| 8 | Learned 0.96-1.04 (L1) beside the "rank-16 floor 0.46-0.52" (Frobenius) (WL94D:109/147/156) | Unmatched norms, flagged at WL94D:147 and corrected in WL95 section 2.2 | As corrected in WL95 | WL94D NO_RELIABLE_IMPROVEMENT unaffected | PRESERVE (correction already recorded) |
| 9 | "Learned S1 operator at the zero-operator level relative to M64" (WL95:244/313/444/491/599; roadmap WL95 entry) | True in flux L1 only; that norm saturates near 1 | Flux L1 0.96-1.04 (zero 1.00, admissible rank-16 0.38-0.60); flux-Frobenius 0.77-0.89 (zero 1.00, bound 0.46-0.52); mass 0.68-1.06; peak 0.11-0.28. M64 itself is biased at delivery | WL95:599 "does not support ... as trained" stands; WL96 design premise | QUALIFY |
| 10 | "Learned S1 operators are non-passive (directional albedo 2.3 / 3.1)" (WL95A:57-58/119; roadmap; WL96:52/136) | Stored as a single maximum for OWN checkpoints | Worst row of the local factor pair; 10-15 % of S1 rows; also WL86 deployed (2.70 / 2.21) and SUM (1.49 / 1.91); sub-unity total energy on CK2; no stability or causal consequence | WL96:136 design caution (fine as a caution) | QUALIFY |
| 11 | "[FACT] No passivity constraint (WL86:41); passivity not measured" (WL95:496) | Superseded | Measured for the local factor in WL95A (OWN) and here (WL86 deployed) | none | PRESERVE |

## 8. Unresolved

- The delivery-optimal patch x 64-bin operator against the path-traced reference (the true class floor on CK2) was never computed, so the size of the
  "avoidable" share inside the class is unknown.
- Whether training can reach the admissible rank-16 level (learnability) is untested beyond WL94D's 1,000-step smoke against a biased teacher.
- No passivity criterion exists for the composed ECRO operator; behaviour of the local factors under recursive reuse is untested.
- E1's training-distribution stage, C1's eight WL94-series checkpoints and C2's paired jackknife were not re-run by this track.
- The sender-error evidence is two seeds of one recipe on one grazing workload.

## 9. Commands and provenance

```
python experiments/audit97/J1/j1_rank16_pareto.py            # -> results/audit97/J1/j1_rank16_pareto.json            (sha256 eee44a4ee89bf279)
python experiments/audit97/J1/j1_rank16_nonneg_construct.py  # -> results/audit97/J1/j1_rank16_nonneg_construct.json  (sha256 9f19b6a912e6ee32)
scp experiments/audit97/J1/j1_wl86_passivity.py LabServer63:rest_x97/J1/
ssh LabServer63 'cd ~/rest_x97/J1 && mkdir -p R/results out && . ~/rest_x95a/env.sh && \
  $PY j1_wl86_passivity.py ~/rest_x97/J1/R ~/rest_x97/J1/out/j1_wl86_passivity.json > j1.log'
# copied back to results/audit97/J1/j1_wl86_passivity.json (sha256 ab638308fb92a566, server copy identical) and j1_wl86_passivity_server.log
```

The first server run used unit patch areas for the exitance ratio (log kept on the server as `j1_run0_area1.log`); the albedo statistics are
area-independent and identical in both runs. The reported run uses `G.PAREA`.
Read-only inputs: `~/rest_ecro86/results/models/ecro_s860{1,2}.pt` (sha256 9d332cd61a7118bd / 4d993121e41dd0bb),
`~/rest_x92a/results/models/{own,sum}_s860{1,2}.pt`, frozen code `~/rest_x96a/code/experiments`.

## 10. Completion check (same session, before hand-off)

Before submitting the structured output I re-read the three J1 outputs and confirmed: file hashes as listed in section 9; the explicit CK2-exact
operator has rank 16 (channel 1), minimum entry >= 0, max albedo 0.975, CK2 field error 2e-18; the rank-16 spectrum (0.520 / 0.458 / 0.461) and the
frontier bounds in section 2.2 match `j1_rank16_pareto.json`; the per-checkpoint albedo rows in section 4 match `j1_wl86_passivity_server.log`
(the log additionally shows the same worst-row non-passivity on other materials, e.g. Lambert-0.7 patches max 1.07-1.62 with mean 0.71-0.76, so the
violation is a property of the learned factor generator, not of the sharp material). WL95A:143-144 and `a95.py:147-152` quotes verified.
No ruling changed.
