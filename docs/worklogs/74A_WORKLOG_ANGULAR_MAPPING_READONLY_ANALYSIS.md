# Worklog — Read-only Angular-Mapping Limit Analysis, WL74A (2026-11-06)

## Origin and scope

**Origin.** On 2026-11-05, while WL74 was running, the user asked (「차라리 네가 현재 ATO에서 Angular Mapping 구조가 Fidelity의 제한점이 될 수 있을지에 대한 제한적 검증을 Code의 Readonly 검증을 통해 분석해볼 수 있겠니?」) for a limited, code-read-only analysis of whether the ATO angular mapping can limit fidelity. This is small unnumbered work, so it is numbered 74A under the letter-suffix rule.

**Scope.**
- Nothing in the repository was changed by the analysis, no server or GPU was used, and nothing was trained.
- Local CPU Monte-Carlo checks written by the agents are non-evidence.

**Method.** A workflow (`ato-angular-limit-readonly`, run `wf_1f8ccaac-382`) did the analysis in three phases:
1. Four independent lenses: algebra, bin frames, historical evidence, transport physics.
2. Two adversarial verifiers per claim: 46 claims, 92 verifications.
3. One synthesis.

**Execution notes:**
- **Interruptions.** The workflow was interrupted three times when the Claude Code session ended. It was resumed each time; completed agents were reused from the journal, but the long physics lens restarted from scratch each time.
- **Usage limit.** A 5-hour usage limit failed 53 agents at about 02:50 KST on 2026-11-06. They were rerun after the reset.
- **No deduplication.** The workflow had no step to merge claims across lenses, so about 8–9 duplicate claims (16–18 verifications) were verified separately. This is a design omission.
- **Not prospective.** WL74 had already been committed when the lenses ran (8528c1b), and several of them read it. Section 7's predictions are therefore not prospective.
- **Keepalive.** At the user's request, a self-ending 2-hour no-sleep process (SetThreadExecutionState) and a 10-minute progress heartbeat were used to keep the session alive.

The synthesis report follows verbatim.

---

# Read-only analysis: can the ATO angular mapping limit K fidelity at boundary sources? (2026-11-06)

## 1. Question and method

**Question.** Can the angular mapping structure of the ATO factorisation limit the fidelity of the learned intrinsic K? The structure here means the global shared A/O, the rank M, the softmax normalisations, the per-port local bin frames, the angle-independence of T, the d = 1 − β coupling and the contrast metric. The question is aimed mainly at boundary, edge, corner and side sources, and asks how much of the observed WL68/WL73 floors this structure could explain.

**Method.**
- **Status.** The study is strictly read-only. No repository file was edited, nothing was committed, no server or GPU was used, and no training ran.
- **Lenses.** Four lenses worked on it (algebra, frames, historical evidence, transport physics). Each lens read the code (cited file:line), reread the WL66–WL73 worklogs and `results/fnto_src73/eval/analysis.json`, and ran small CPU numpy computations.
- **Monte Carlo.** The lenses wrote their own first-exit Monte Carlo walkers of the same slab. They reproduce the interior reference port 2085 within MC noise, and the WL73 labelled fixed-A/O oracle within about ±0.01. They also ran signed and nonnegative CP/ALS relaxations, Tucker controls, NNLS cone tests and symmetry-image fits on the coarse (240 × 16) and native response.
- **Adversarial review.** Two independent verifiers attacked every claim, re-deriving the algebra and re-running the numbers. Some verifiers wrote their own code.
- **Status of the numbers.** All MC values are non-evidence. They come from the rest geometry only, are mostly coarse-space and single- or two-channel, and are point estimates. They are not the WL59 gate statistic (worst-channel upper 95 % bound).

**Provenance caveats.**
1. Several scratch scripts that produced cited logs were later overwritten (`cpfit.py` lost `fit_cp`; `mc.py` changed its signature). The affected numbers were reproduced from the stored `.npz` outputs with independent re-implementations, but the original generators cannot be re-run as cited.
2. **WL74 was already complete at HEAD (commit 8528c1b; `docs/worklogs/74_WORKLOG_ANGULAR_REPRESENTABILITY_AND_TRANSFER.md`), contrary to the task brief.** The physics lens and most verifiers read it. So Section 7 is **not** a set of prospective predictions. It records what was predicted and how each item stands against the committed WL74, and separates out the few checks that still touch unexamined WL74 data.
3. A verifier stopped a scratch process it had not launched (another agent's run of a scratch script). Nothing in the repository was affected.

---

## 2. What the code proves about the angular operator class

All the statements in this section are exact and follow from the code.

### 2.1 Operator form

`family68.py:44` and `ato74.py:22` both compute `einsum('na,rac,njrc,roc->najoc', d, A, T, O)`. In that expression:
- A = softmax(in_logits, dim r) and O = softmax(out_logits, dim o), at `family68.py:6-7,38,108`, `runtime.py:65`, `r57_model.py:25-26,67` and `ato74.py:36`.
- A and O are model-level parameters, one per RGB channel, with no port, face or geometry input.
- T[i,j,r,c] ≥ 0 has no incoming-bin index. Its mass is capped: Σ_j T ≤ 1, from the sigmoid mass in `runtime.py:58-62` and `project_capped` in `family68.py:114-122`.

Consequences:
- **(P1) Per-block cone.** For each (source i, receiver j, channel c), the budget-normalised 16 × 64 block is K/d = Aᵀ diag(T_ij) O. Every block is a nonnegative combination of M fixed rank-1 atoms A_r ⊗ O_r. In local-bin index space these atoms are identical for every source, receiver, face and geometry. Because the frames differ by face (Section 3), the same atom is a different world-angular pattern on different faces.
- **(P2) Per-source CP form.** For each source, Z_i = S_i/d_i is a nonnegative CP tensor over (a, j, o) of rank ≤ M. The fine16 and coarse aggregations only sum over O and T (`family68.py:42-43`), so the bound also holds in those spaces.
- **(P3) Convex incoming mixing.** Σ_r A[r,a] = 1, so Z_i[a] = Σ_r A[r,a] Y_ir with Y_ir = T_ir ⊗ O_r. Each incoming bin's image is therefore a convex combination of M per-source images, with globally fixed barycentric weights.
- **(P4) Centred incoming span.** Aᵀ1_M = 1_16, so H Aᵀ 1 = 0 and rank(H Aᵀ) ≤ min(M−1, 15). The centred incoming contrast of every source lies in one subspace shared by all sources, of dimension 7 at rank 8 and 15 (unrestricted) at rank 16. At M = 1 the predicted contrast is identically zero.
- **(P5) Joint per-cell restriction.** For each (source, coarse cell, channel), the centred incoming × outgoing block is H Aᵀ diag(t) O16 with t ≥ 0. It lies in a global span of dimension M out of 240 centred dimensions (256 uncentred). Rank 16 raises this from 8 to 16 dimensions; it does not remove it. Incoming discrimination and outgoing lobe choice share the index r: whichever mixture carries incoming bin a also fixes the outgoing lobe.
- **(P6) Outgoing hull.** At every (i, a, j), the collided outgoing shape lies in conv{O_r[:, c]} in the receiver's local frame. The ballistic part B lies outside this constraint.
- **(P7) Exit totals.** The per-bin collided total is Σ_{j,o} K = d_a Σ_r A[r,a] m_r ≤ d_a max_r m_r ≤ d_a, with m_r = Σ_j T. Because A is global and convex, the cap can bind even when every physical bin total is below d (Section 6, mass cap).
- **(P8) The d coupling.** d = 1 − β_model comes from a fixed-seed MC quadrature: q = 256 rays per (port, bin), seed 57501 (`r57_geometry.py:75-76,80-94`). d is used in prediction, fit targets and evaluation (`ato74.py:67,160,169`; `eval73.py:44`; `fit73.py:48`; `eval74.py:34`; `eval68.py:53`). In the contrast term the model's d cancels exactly, because the prediction is divided by the same d it was multiplied by (`family68.py:89`, `ato74.py:33`). The target is H(S/d_model) (`family68.py:60`), while the walker reference S carries the physical d_true. So the factor d_true/d_model enters only on the target side, as a per-source, per-bin factor.
- **(P9) Contrast metric.** con = |H D⁻¹(Ẑ − Z)| / |H D⁻¹ Z| on the coarse response (`family68.py:53,60-61,89-90`; `eval68.py:52-60`). In the model, the contrast is Σ_r (A[r,a] − Ā_r) Y_r. It is built from the same images as the incoming-mean response Σ_r Ā_r Y_r. The incoming-mean error is invisible to this term. The objective weights contrast equally with the mean of the three abs terms (`family68.py:91`).
- **(P10) Nesting.** G0 ⊂ G1 ⊂ G2 exactly. Duplicating the atoms with halved A and duplicated T reproduces the rank-8 operator to ≤ 1e-12 (`test74.py:17-27`).
- **(P11) WL72 symmetry term.** The WL72 L_sym compares angle-integrated labels (`sym72.py:31-37`; `fit72.py:39-41,67,75-77`). Its labels sum over both a and o and are permuted over ports only. O cancels because rows sum to 1, and A enters only through Σ_a d_a A[r,a]. A and O were frozen (asserted bit-identical, `fit72.py:79,95`). WL72 therefore carries no information about angular equivariance.

### 2.2 Rank-limited constraints versus structural ones

| Constraint | At rank 8 | At rank 16 | Any global A/O? |
|---|---|---|---|
| Centred incoming span ≤ M−1 (P4) | binds: 7 of 15 | lifted (15 of 15). L15 = 0 holds by construction, so it is not evidence | n/a |
| Joint per-cell (a,o) span = M (P5) | 8 of 240 | 16 of 240 | lifted only as M approaches 16 × 64 with indicator atoms |
| Orientation copies (Section 3) | spends mixtures | spends fewer, still costs | finite multiplier for exact slab symmetries (≤ 8 bin symmetries) |
| Diagonal pairing: one incoming profile ↔ one spatial map T_r ↔ one global lobe O_r | binds | binds less | lifted only in the impractical limit above. The A → I limit (not reachable under softmax) ties each incoming bin to a single global lobe |
| T has no incoming index (P3) | structural | structural | structural |
| Atoms independent of frame, face and position | structural | structural | structural |
| Normalisation Σ_j K[i,j,a,o] ≤ d_a (Aᵀ O)[a,o], with Aᵀ O global and row-stochastic | holds | holds | **holds for every M** |
| d from q = 256 quadrature (P8) | fixed | fixed | not a property of A/O |

A verifier correction applies here. The cone and pairing restrictions are not intrinsic at every rank: they disappear with about 16 × 64 indicator atoms. Only the normalisation bound in row 7 holds at every M. In this slab it binds mildly for red, where the collided exit fraction is about 0.99 (envelope sum about 1.08 over boundary sources), and not for blue. Its effect is small (per-source marginal deviation 3–5 %, 9–10 % at a corner).

---

## 3. Bin-frame findings

### 3.1 Frames

`p53_geom.py:95-102,112` (voxel_boundary: e1 along axis (ax+1)%3, swapped for right-handedness) and `tri_frames` (b = n × t, `p53_geom.py:179-194`) give the frames below. Refinement inherits port_tangent (`c54_casp.py:81`). The frames were recomputed independently by several verifiers.

| Face | t | b | n | Slab vertical axis |
|---|---|---|---|---|
| f0 (−x) | +z | +y | −x | local u |
| f1 (+x) | +y | +z | +x | local v |
| f2 (−y) | +x | +z | −y | local v |
| f3 (+y) | +z | +x | +y | local u |
| f4 (bottom) | +y | +x | −z | t/b transposed relative to the top |
| f5 (top) | +x | +y | +z | — |

### 3.2 Bins

- **Equal projected solid angle.** The bins are concentric-square (Shirley–Chiu / Malley), so each gets exactly π/N² of projected solid angle by construction. The MC check gives 0.996–1.004 at N = 4 and 0.991–1.013 at N = 8.
- **Symmetry.** The grid is exactly equivariant under local D4 (purity 1.0 at N = 4 and N = 8), and `bin_parent` nests 64 bins into 16.
- **Toward-edge directions sit on bin boundaries** (u = 0.5 or v = 0.5 for even N):
  - outer ring: az 0 → {7, 11}, 90 → {13, 14}, 180 → {4, 8}, 270 → {1, 2};
  - inner ring: {6, 10}, {9, 10}, {5, 9}, {5, 6}.
  - So the top +x edge is {7, 11} seen from f5 but {13, 14} seen from f1, and the bottom +x edge is {13, 14} from f4.
- **Polar resolution.** The 16-bin grid has rings at 30° and 90°. Along axis and diagonal azimuths, all of 30–90° falls in one bin. At general azimuths the cells do partly split by polar angle (corner bins have mean θ 58.9° against 50.8° for mid-side bins). The 64-bin grid lumps 48.6–90°.
- **This resolution limit cannot enter any measured floor.** The reference (`cell_source` / `bin_sample`), d and all metrics use the same bin-averaged incidence.

### 3.3 Symmetry action

- A slab symmetry M induces Q = F_{f'}ᵀ M F_f, an exact local D4 permutation with det Q = det M.
- **12 of the 16 slab symmetry ops are face-dependent**, inducing three distinct local maps each. The uniform ones are id, sw, r180z and aswz. The algebra lens said 13; the verifiers corrected this because id was miscounted.
- Example: r90 induces r90 on f0/f5, r270 on f1/f4 and id on f2/f3.
- Even under any right-handed outward-frame convention, the top face alone already carries the full local D4. Edge and corner sources within one face therefore need all orientations whatever the convention.
- The part this specific convention adds is limited to the side faces: the thin axis is t on f0/f3 but b on f1/f2.

### 3.4 Quantitative orientation results

These are MC proxies, verified by two verifiers each unless noted.

| Test | Result |
|---|---|
| **G0 on exact symmetry images** (16 images each of TBE 3439 and SR 527, R channel, true d, converged to 30k FISTA iterations, deterministic) | con 0.223–0.276, abs_coarse 0.095–0.153. Worst vs best orientation is 31–35 % of the con error energy and 2.6× the abs_coarse energy. The top-face rotations alone span 0.229–0.274. The WL73 identity orientation is mid-range. |
| **Control:** A/O symmetrised over the 8 local permutations | The spread vanishes to 5 decimals, but con rises to about 0.87. The orientation dependence is tied to how a per-bin global basis encodes contrast. |
| **Interior equivalents** under fixed WL68 A/O (1699, 1123, 1692, 2400, 1823; identical contrast fraction 2.83–2.85 % and identical relaxed floors) | con 0.150–0.241. Within one channel the spread is 0.07–0.08 (0.09 is a cross-channel figure). Bin-reflecting 1699 reproduces the others' floors. The best orientation matches WL68 fit port 1123. 2400 shares 1699's face frame and still scores 0.200 (in-cell offset (1.25, 1.25) vs (1.25, 8.75)), so "frame orientation" really means orientation of the response relative to the coarse cell, in the bin frame. |
| **Boundary sources, best of 8 orientations, fixed WL68 A/O** | TBE 0.252/0.259/0.273, TBC 0.261/0.256/0.271, SR 0.231/0.235/0.256, against 0.146–0.164 for interior cell-corner sources at their best orientation. In native frames the boundary excess over interior corners is 0.01–0.14. |
| **Shared refit over rotated copies** (aligned = data rotated back into one frame) | At M = 8: TBE 0.17–0.19 raw vs 0.064–0.075 aligned (single-source fit 0.073); SR 0.14–0.15 vs 0.077–0.082. At M = 16: TBE 0.095 vs 0.029–0.034. Aligning removes the whole multi-orientation excess. |
| **Frame-convention share** (side-face t′ = +z) | SR4 0.142 → 0.101 (about half the SR excess). Incoming part of the {0,3}/{1,2} split about 0.035 (SR f2+f0 0.129 vs f2+f1 0.094). TBE4 is unchanged (0.171 vs 0.174), because all four copies are on f5. |
| **Outgoing span vs frames** | The free signed rank-8 floor changes by ≤ 0.001 between native, D4-image and canonical frames (≈ 0.026–0.039, against a noise of about 0.02–0.03). Frames add no linear-span pressure. |

---

## 4. Physics expectations

Each item is labelled by the kind of evidence behind it.

1. **Exact factorisation** (derivation). For g = 0 and index-matched boundaries, the collided first-exit kernel factors through the first-collision location: Z_i[a](j, o) = ∫ p_{i,a}(x) G(x; j, o) dx. The incoming bin enters only through where the first collision lands: depth about cos θ · 1 mm and lateral offset about sin θ (cos φ, sin φ) · 1 mm. The natural intermediate index is location, not an angular class. This is consistent with ATO's angle-free T, but the angle dependence must be carried by A.
2. **Location-integrated a × o is nearly rank 1 everywhere** (numeric check; corrected). σ₂/σ₁ = 0.016–0.017 at interior ports and 0.020–0.023 at edge, corner and side ports; σ₃/σ₁ ≤ 0.012. "Far from edges" is not required. The outgoing shape is near-Lambertian. At grazing incidence the measured shift is 0.94–0.95 inner / 1.01–1.04 outer (one verifier) against the claimed 0.90–0.92 / 1.03–1.09. The single-scatter fraction is 0.16 at normal incidence and 0.24 at grazing, against the analytic 0.153 at μ₀ = 1.
3. **The coarse grid turns footprint shift into contrast** (numeric check, two verifiers). The metric aggregates into 1 cm cells, so the azimuth-dependent lateral shift becomes incoming-bin contrast when the source sits near a cell boundary.
   - |HZ|/|Z| (G channel): about 0.05 at centre sub-cells, about 0.13 at edge sub-cells, 0.17 at corner sub-cells.
   - At a corner sub-cell the own-cell share swings 0.50–0.73 across incoming bins, while 27–50 % of the collided exit leaves the cell; the centre sub-cell keeps 0.78–0.85.
   - At port (2.5 mm) resolution the centred incoming structure is identical for centre and corner sources. The difference is purely the grid.
   - At a corner, the absolute contrast residual of the fixed-A/O fit grows about 8× while the signal grows 3.4×, so corner sources are genuinely harder to fit, not just rescaled.
   - This matches WL74's per-port G0 classes (corner 0.237 / 0.101, edge 0.13 / 0.065, (1,3) 0.113).
4. **Per-source incoming rank** (numeric check). At the coarse metric resolution the centred incoming dependence has rank about 3–5. It is a polar/depth mode plus two shift modes (sin θ cos φ, sin θ sin φ) plus about two second-order terms; the span of x, y and r² holds 94.5–98.8 % of the energy. At port resolution the rank is about 6–8.
5. **Edges are not special once d is divided out** (numeric check, from the refutation of physics C10/C6).
   - Dividing by d is physically appropriate: |HS|/|S| → |HZ|/|Z| goes 0.151 → 0.111 at an edge and 0.204 → 0.147 at a corner.
   - An interior port with β = 0 at the same sub-cell offset as the edge port has *more* residual contrast (0.127–0.138 vs 0.108–0.131). Interior corner sub-cells (0.171) exceed the slab corner (0.147). The edge influence fades within one port.
   - Edge outgoing tilts match the interior's displaced-receiver tilts to a few %. About 85–90 % of the edge/side shape structure lies in the cone of interior shapes (NNLS residual 1.4–2.4 % rms). The genuinely edge-specific part is truncation at the source port itself, about ±5 %.
   - What does depend on the edge is the side-exit fraction, which varies with incoming azimuth: 0.25–0.41 at an edge port, 0.37–0.56 at a corner, 0.55–0.70 on the top side row. That is spatial × incoming coupling (handled through T and A), not a demand on O.
6. **Grid-boundary confound** (data). The 1 cm cells are aligned with the slab edges (`r57_geometry.py:46`, `KO.refined(m, V, 4)`). Every edge-row source is therefore also 1.25 mm from a coarse-cell boundary, and the WL73 TBE/SR/TBC ports themselves sit at cell-corner sub-cells. Boundary-versus-interior comparisons in WL73 groups are confounded with sub-cell position.

---

## 5. Historical evidence

**For an angular-structure limit**
- **WL73 training floor.** The new boundary training sources fit to 0.110 / 0.119 / 0.270 in D1, A and B, equal to the labelled fixed-A/O oracle (0.111 / 0.117 / 0.270) source by source within about 0.003.
  - An independent CPU recomputation (fixed WL68 A/O = `net_all_5902`, free T ≥ 0) reproduces the per-source contrast to within −0.013 / +0.006 at 8 distinct sources. It is systematically slightly low at boundary sources, as a relaxation should be.
  - Down-weighting the abs term 100× lowers boundary contrast only to 0.25–0.27, so the floor is not a weighting artefact.
  - This confirms WL73's own attribution. It is not a new finding.
- **WL68 (WL59 basis).** The admissible floor at frozen WL59 A/O is 0.736–0.757 on rest, and the network fits land on it. Freeing A and O gives 0.57–0.61 (net); fully free fits give 0.13–0.16 (net_all) and 0.12–0.14 (FCoef).
- **WL66 (WL59 basis).** The learned WL59 A discards ρ 0.37–0.55, against a relaxed floor of 0.025–0.027.
- Items 2–3 concern the superseded WL59 basis, not the WL68 basis fixed in WL71–74.

**Against the angular basis being the main held-out limit**
- The fixed-WL68 oracle passes at interior held-out ports: WL68 0.146–0.150, WL71 FIN 0.175, WL73 INT_DED 0.178, REM_NEAR 0.121. It also passes at the side-edge UNS group (0.237) and at WL72 CONF (0.21 pooled). WL72 CONF includes side and corner ports, so it is not an interior group. The oracle fails only at top/bottom boundary-edge groups (BND_NEAR 0.268, BND_FAR 0.267, training sources 0.270).
- Learned held-out contrast is 2.5–6.5× the oracle: WL72 0.55–0.87 vs 0.21; WL73 BND_FAR 0.87–1.30 vs 0.267; UNS 1.02–1.53 vs 0.237.
- Learned models and oracle share the same A/O and exact d, and the metric is linear. The triangle inequality therefore bounds the T-map part of the error at no less than the gap (about 0.6–1.0 at BND_FAR), with no orthogonality assumption.
- WL66's "frozen A explains 19–41 %" is of error *energy*, and WL68's admissible floors supersede it.

**Descriptive energy shares** (oracle/learned)² from WL73, worst-channel (channel-matched in brackets). These are ratios, not an exact decomposition. The oracle minimises a combined per-source objective on one replica, so the cross term is unsigned.

| Group | Share |
|---|---|
| TRAIN_NEW | about 99.5 % |
| BND_NEAR | 27–42 % (26–56 %) |
| INT (INT_DED proxy; no INT_ALL oracle exists) | 6–56 % |
| BND_FAR | 4–9 % (4–13 %) |
| UNS | 2–5 % (2–10 %) |
| REM_NEAR | 1–19 % (1–22 %). The original claim's 1–6 % omitted B seed 7202 at 0.282. |

**WL74 (committed; read, not guessed)**

| Basis | TRAIN_NEW abs / fine16 / con | BND_DEV con | TRAIN_KEEP_rest con |
|---|---|---|---|
| G0 | 0.111 / 0.117 / 0.270 | 0.258 | 0.188 |
| G1_r0 | 0.113 / 0.127 / 0.222 | 0.231 | 0.166 (abs +31 %) |
| G1_r1, r2 | worse everywhere | | |
| G2 (r0–r2) | 0.084–0.098 / 0.096–0.119 / 0.129–0.137 | 0.154–0.157 | 0.088–0.108 |

- G1 restart spread is 120 %. G2 halves contrast in every context; abs falls 10–30 %.
- G0 contrast at interior corner sub-cells is 0.237, the same as at boundary ports.
- Verdict: class C + D. Stage 3 did not run.

---

## 6. Mechanisms that could limit fidelity at boundary sources (upheld by ≥ 1 verifier)

Ranked by measured relevance to the boundary floors. "a/b" is the number of verifiers, out of 2, who did not refute the claim.

1. **The shared low-rank CP pairing under angle-independent T, acting on grid-position-dependent footprint shift.** Each mixture ties one incoming profile to one spatial field and one global outgoing lobe.
   - Supporting claims: algebra C1 (2/2), C2 (2/2), C6 (2/2), C9 (2/2); evidence C1 (2/2), C3 (2/2); physics C1 (2/2), C4 (2/2), C5 (2/2), C8 (2/2); frames C11 (1/2, corrected).
   - Per source, a rank-8 nonnegative A·T·O with its own A and O reaches contrast 0.07–0.11 (split-half confirmed). The CP form per source is therefore not the limit.
   - Sharing A/O is the cost:
     - One A/O over 13 boundary sources (signed): about 0.145–0.15 at M = 8 and 0.07–0.08 at M = 16.
     - Over 4 boundary + 4–5 interior sources (nonnegative, nonconvex, unconverged): 0.15–0.25 at the boundary.
     - WL74 admissible full-J G1_r0: 0.22–0.23.
   - The relaxations understate the admissible floors by about 0.05–0.08 at the boundary and 2–3× at the interior (algebra C11, 1/2; refuting verifier).
   - **Toy share.** A signed shared rank-8 fit reproduces about 55–65 % of the boundary contrast floor in magnitude (about 30–45 % of the squared error) and most of the interior corner floor (nonnegative toy 0.19–0.21 vs 0.237).
   - The toy predicts that rank 16 halves abs; WL74 abs barely moves. **This mechanism explains a large part of the directional-contrast floor, not the abs floor.**
   - It is not boundary-specific. It tracks sub-cell position.
2. **Specific values and orientation non-equivariance of the fixed WL68 A/O.**
   - Supporting claims: frames C8 (2/2), evidence C5 (1/2), frames C4 (2/2, corrected), frames C7 (1/2, corrected).
   - With a fixed basis, physically identical sources differ by 0.04–0.05 con between best and worst orientation (31–35 % of the energy) and by 0.07–0.08 for interior equivalents.
   - WL68 A has effective incoming rank about 5, not 7: σ 0.59/0.50/0.46/0.10/0.04 then ≤ 0.001 (evidence C9, 1/2). The top-5 span covers sin2φ only partly (residual 0.41–0.48) and lacks cos3φ and inner-bin azimuth.
   - **Corrected magnitude for that A deficit:** with free nonnegative outgoing coefficients, the fixed A alone bounds boundary contrast at about 0.11–0.19. That is about 25–45 % of the boundary contrast error energy. Most of the rest comes from coupling A to fixed O through scalar T.
   - Cross-face receivers (receivers on a face adjacent to the source's) carry 59–75 % of the squared native G0 error at only 15–27 % of the energy. Same-face receivers reach 0.05–0.07 under the G0 O cone versus 0.15–0.17 cross-face. A free nonnegative 8-profile cone fitted to the boundary sources reaches about 0.04, so this is G0-specific shape, not rank.
   - The orientation mechanism explains the *spread* around the fixed floor, not the floor itself. The identity (WL73) orientation sits mid-range.
3. **Orientation diversity under a shared refit.**
   - Supporting claims: algebra C7 (2/2), frames C4 (2/2), frames C7 (1/2).
   - Rotated copies of one physical source raise shared M = 8 contrast from about 0.07 to 0.14–0.19. Aligning the data removes the whole excess, and about half the SR excess is due to the side-face tangent convention.
   - A transfer test with A/O fitted on the WL73-like orientations gives 0.18–0.19 on other orientations against 0.14–0.15 on the trained ones (M = 8). At M = 16 it gives 0.125–0.136 against 0.072–0.078. The test set is mostly exact images of training sources.
   - This mechanism matters for refit or learned A/O and orientation transfer. It is minor for the fixed-WL68 floor, which is roughly orientation-uniform in level.
4. **Incoming span alone.**
   - Supporting claims: frames C6 (2/2), evidence C2 (2/2), physics C7 (2/2), algebra C5 (1/2).
   - Relaxed rank-7 floors per source are about 0.014–0.024 (low-biased).
   - Shared across sources: 0.034–0.049. With symmetry images and model d: 0.076–0.079.
   - The fixed G0 A span floor is 0.09–0.11 pooled at boundary sources. That is about 13–17 % of the G0 contrast error energy pooled and up to about 26 % for single images.
   - A Tucker(8,8) control with shared A/O and a free core gives 0.029, matching L7. So the step from about 0.027 to about 0.075 (signed, interior) is the diagonal CP pairing, not incoming or outgoing rank.
   - Minor.
5. **d = 1 − β quadrature error** (algebra C8 2/2; frames C9 2/2; evidence C7 2/2; physics C9 2/2, corrected to minor).
   - Per-bin |Δβ| rms is 0.008–0.013 at TBE/TBC/SR (z ≈ 1, i.e. pure q = 256 noise), up to 0.024 at the training ports and 0.038 at confirmation ports. It is about 0 at SM and interior ports.
   - Effect on the fixed-A/O oracle: con +0.004 pooled per channel (TBC +0.009 to +0.014; symmetry-image ports mean +0.007 to +0.010, up to +0.033); abs ≤ 0.005. That is about 3 % of the pooled boundary contrast error energy and 5–13 % unabsorbed at edge or corner sources.
   - It is identical in every arm. Its fractional share grows in arms with lower other errors.
6. **Contrast-metric normalisation** (evidence C8, 2/2, corrected).
   - Contrast signal is 1–3 % of the coarse response energy. con ≤ 0.25 bounds the centred error at about 1.2–4.3 % of |Z|.
   - Observed con/abs is only about 2–3, and the worst contrast floors occur where |HZ|/|Z| is *largest*. This is not "amplification" of a small misfit.
   - At fixed A/O, 77–94 % of the abs-coarse residual is incoming-mean. A contrast-free refit cuts it 2.2–5×, so much of the abs floor at fixed A/O is induced by the contrast/abs trade-off through shared T, not by O-lobe inadequacy alone.

**How much of the observed floors**
- **WL73/G0 boundary training floor (0.27 con; 0.11 abs):** essentially all of it is the fixed-A/O ATO class (A/O plus T structure, cap and d). Indicative partition, not additive and not certified:
  - incoming span ≤ about 15 %;
  - d about 3 % pooled;
  - the A-span deficit alone about 25–45 %;
  - the remainder is A–O coupling through scalar T under global sharing.
- **After an admissible shared refit:** rank 8 still leaves 0.22–0.23 (WL74), and rank 16 leaves 0.13–0.17. These are the measured structural floors of global A/O at each rank on this population.
- **Learned far transfer:** the angular structure bounds BND_FAR at about 0.27 (G0) or about 0.16 (G2), against learned 0.87–1.30. About 70–80 % lies above the bound and is spatial. Near supervision (BND_NEAR) the angular share is about 27–42 %.

---

## 7. Predictions for WL74, stated against the committed results

The brief asked for prospective predictions. **WL74 was already committed, and several lenses and verifiers read it, so none of these predictions are independent of its results.** Each prediction is recorded as stated and then checked.

| Lens prediction | WL74 outcome |
|---|---|
| G0 reproduces the WL73 oracle (about 0.11 / 0.12 / 0.27) | Matches (0.111 / 0.117 / 0.270) |
| G1 boundary contrast falls below G0 but stays near or above about 0.15 | Matches in direction; actual 0.22–0.25, worse than the signed relaxations |
| Interior non-regression at M = 8 is not threatened (algebra C5) | **Contradicted:** G1_r0 interior abs +31–35 %, rule failed |
| Contrast remains the binding metric for boundary groups in every arm (algebra C4) | **Contradicted:** abs_coarse binds in every G1 and G2 arm |
| G2 boundary contrast about 0.06–0.11 (signed or toy) | **Too optimistic:** 0.129–0.167 |
| G2 roughly halves contrast in every context, interior included | Matches (×0.47–0.75); stated after reading WL74 |
| The floor is not boundary-specific and tracks the 1 cm grid sub-position | Matches (corner sub-cells 0.237); stated after reading WL74 |
| Refit O helps cross-face receivers more than same-face receivers | Not checkable: no per-receiver-face breakdown was reported |
| A74 transfers beyond its representability gain | Not checkable: Stage 3 did not run |

These checks still touch WL74 artefacts that have not been examined. They are listed as predictions, not as proposals:
- **Per-receiver-face breakdown of G0 vs G1/G2.** Expect the largest relative native/fine16 gains at cross-face receivers.
- **Spread across physically equivalent orientation images.** Expect G2's spread to be narrower than G0's 0.04–0.05 con but not zero.
- **Rescoring any arm with true d.** Expect con to fall by about 0.004 pooled and by about 0.01 at TBC, with a larger relative share for G2.

---

## 8. What a read-only study cannot decide, and refuted claims

### Cannot decide
- The certified rank-8 shared optimum (WL74 restart spread 120 %). Every relaxation here is a nonconvex achieved value, not a bound.
- The minimal rank between 8 and 16.
- Whether a rank-16 learned K would transfer better or worse.
- Behaviour on deformed or other geometries, glossy or heterogeneous media, and the full native/fine16 objective. The coarse proxies cover rest geometry only, 1–3 channels and 4–13 sources.
- A causal test that the grid-relative footprint shift is what drives WL74's per-port classes. It is consistent with the data but not isolated.
- Whether a mismatched basis makes the geometry→T target harder to learn (evidence C12 was refuted as stated; see below).

### Refuted claims (0/2 upheld), with the verifiers' replacements, so they are not re-raised

| Claim | Status | Replacement |
|---|---|---|
| algebra C4 | Contrast metric amplifies error 6–21×, and contrast stays the binding metric in every arm | Observed con/abs is about 2–3; abs binds in G1 and G2 |
| algebra C10 | The mass cap does not bind | The cap is active in every WL74 basis (verifier read `T_max_colsum = 1.0000000000000004` from `results/fnto_ato74/stage1/scores.json`; not re-checked here). In local WL59-basis interior fits it costs 1–2.5 % of J, and 5–21 % (R/G) at the source nearest a side. It binds because misfit lobes push m_r above 1, not because d is low. Its size at boundary sources with WL68/74 bases is unmeasured. |
| frames C3 | Face-pair-specific specular/transmission bin maps | The collided operator (g = 0, B separate) has no such continuation. Its a–o interaction is position-mediated, same-face "retro", and equal for interior and boundary. |
| evidence C4 | The boundary floor comes "nearly all from the specific WL68 bases, not ATO" | Global sharing is part of ATO. Shared refits recover only about 25–79 % of the per-source-reducible energy, depending on the population. |
| evidence C12 | Basis mismatch makes T less learnable | Translation-covariance within a face rules this out for interior failures. TB, trained under a mismatched basis, is smooth. Only an untested symmetry-image variant remains. |
| physics C6 | Edges demand extra asymmetric O shapes | The interior already needs stronger oriented shapes. |
| physics C10 | Edge-specific truncated-collision residual | No edge excess at matched sub-cell position. |
| physics C11 | Amplification explains the floors | Floors grow with \|HZ\|/\|Z\|. |
| physics C12 | "Well beyond rank 16" needed | The toy shows shared-16 ≈ per-source-8 on contrast; WL74 G2 meets the gates as an oracle. Orientation mixing costs no more than position mixing. |

Partially refuted, kept with corrections:
- **algebra C3** (outgoing hull material): interior evidence argues against it. It survives only as the cross-face, G0-specific finding (frames C7).
- **physics C2** ("rank 16 removes the restriction; only coupling remains"): the joint per-cell span persists at rank 16.
- **physics C3** (WL74 support via an a×o argument): the relevant coupling is a×j, and the WL74 classes cited are edge sub-cells, not cell centres.