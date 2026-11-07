# WL77A Coordinator Decision: Candidate Selection (2026-11-07)

**Author.** The coordinator (main session). I checked the adjudication draft (`adjudication_draft.md`) against the workflow files, the code and three non-evidence numerical checks of my own.

**What this does not do.** It adopts nothing and authorises no implementation.

## 1. Decision

**No candidate is selected (NONE). WL77A stops at Part D**, under the brief's rule: "if no candidate survives the review, STOP and report the negative discovery result".

| Candidate | Central mechanism | Decisive objections (all UPHELD) | Coordinator assessment |
|---|---|---|---|
| H1 CQR, Canonical-Quotient Relative Kernel | An exact slab-symmetry quotient, plus a learned angular-spatial pair kernel in canonical relative coordinates | A1, B1, B3, A7, B2 | **Rejected.**<br>• Its only mechanism-derived transfer is the orbit and translate identity, which any de-identified equivariant model, including a W+UV-Q control, shares.<br>• Out-of-orbit contexts rely on MLP interpolation of multiply scattered source-side functions from about 4 boundary configurations. H1 itself predicts this fails.<br>• The quotient / frame averaging is prior art (Puny 2022; Dym 2024).<br>• Its state (about 1.5–2.1 GB with frame multiplicity) sits at the cap. |
| H2 RBG, Reciprocal Boundary-Layer Green Form | Exact first-collision / escape projections Π with a learned scalar, source-free element-pair resolvent Γ (Πᵀ Γ Π) | A4, A5, A7, A12, B6, B7 | **Rejected.**<br>• Projecting the first collision onto 2.5 mm hat functions (h > mfp) removes the incidence footprint shape it needs.<br>• The near-diagonal Γ block at unseen plane configurations is identified by smoothness only.<br>• The SS cubature is aliased.<br>• The pose-change cost exceeds the caps.<br>• The mechanism is covered by Neural Green's Functions / Element Learning / DeepRTE. |
| H3 LCR, Learned-Closure Volumetric Resolvent | A learned local closure of the collision generator, followed by an exact collision-probability solve | A8 (Horn A), B4, B5, B6, A9, A13 | **Rejected.**<br>• Its transfer is carried by a published classical CPM / first-collision-source solve; the learned closure is likely vacuous (below seed spread).<br>• Admissibility as a learned FNTO K is an untaken human decision (contract l. 17 / 86).<br>• The apply meets the cap only with an unbuilt kernel.<br>• It needs a known scalar kernel, so it does not extend beyond g = 0, homogeneous, index-matched media. |

**How the evidence was weighed.**
- I did not count votes. Every UPHELD ruling I relied on cites code or arithmetic, and I spot-checked the ones that decide selection: `r57_geometry.py:58-66` (absolute-position node features), `ports76.py:1-5` (point-group exclusion only), `gate77.py:45-48` (noise estimate).
- The three refuted sub-arguments do not rescue any candidate:
  - the RSK drift precedent;
  - Horn B as an impossibility;
  - the top corners being unsupervised at pt0.
- Two rebuttals stay unresolved (H1 batched-GEMM apply; H3 custom stencil kernel). Both are cost questions, not transfer questions.
- No candidate breaks a hard invariant. Linearity, positivity, passivity and K/C consistency hold for all three. All three fail on the transfer mechanism, the information or the novelty, which are the criteria the brief says no other criterion may substitute for.

## 2. Coordinator verification checks (non-evidence; `experiments/fnto_arch77a/coord_checks*.py`, `results/fnto_arch77a/`)

### C1. SS-quadrature noise in the WL76 / WL77 floors (objection A6; neural specialist F7–F8)

**Method.** Gate77 projection floors for training sources s0 and s50, with the single-scatter term SS computed by the WL76 quadrature at increasing q.

| SS | ring | FIA-AZ (cos, cos2 of φ_o − φ_r) | free phase, order ≤ 2 |
|---|---|---|---|
| q = 1024, WL76/77 seeds (the WL77 record) | 0.453 / 0.474 | 0.386 / 0.408 | 0.353 / 0.370 |
| q = 1024, independent seeds | 0.449 / 0.456 | 0.379 / 0.392 | 0.346 / 0.356 |
| q = 4096 | 0.301 / 0.307 | 0.212 / 0.226 | 0.188 / 0.195 |
| q = 16384 | 0.242 / 0.242 | 0.120 / 0.140 | 0.094 / 0.105 |
| q = 65536 | 0.227 / 0.219 | 0.084 / 0.100 | 0.051 / 0.055 |
| **extrapolated noise-free** (residual² = true² + c/q) | **0.22 / 0.21** | **0.067 / 0.083** | **0.021 / 0.020** |

- The pure q = 1024 SS noise is 0.45 / 0.48 of the M norm.
- **Consequences.**
  1. The WL77 Stage 1 statistic was dominated by the noise of the q = 1024 SS used to form M. The replica half-split noise estimate could not see it.
  2. With noise-free SS, FIA-AZ / ring is about 0.30–0.40, below the frozen 0.5 gate.
  3. The WL77 interpretation "the remaining in-ring structure is order ≥ 3 azimuthal" is contradicted: free-phase order ≤ 2 reaches about 0.02.
  4. The WL76 post hoc reading "native error ≈ ring floor 0.45" mixed a true ring restriction (about 0.21) with SS noise (about 0.4).
- **Status of the old records.** The recorded WL77 verdict (STOP by the frozen rule) and WL76's measured INSUFFICIENT FIT are historical facts. Only their interpretation is corrected here, append-only.

### C2. Error from SS-quadrature noise alone in an FIA-type prediction (q = 1024 deployable SS)

**Method.** Prediction = (reference − SS_true) + SS_q1024, i.e. a perfect learned multiple-scatter part, with SS_true taken as 4 × q = 65536. Point relative L2 of the replica mean, not the eval68 upper bound.

| Source | native | fine16 | coarse | coarse contrast | reference replica noise (native / contrast) |
|---|---|---|---|---|---|
| s0 interior | 0.267 | 0.135 | 0.069 | **1.24** | 0.001 / 0.006 |
| s50 boundary | 0.276 | 0.138 | 0.096 | **0.57** | 0.001 / 0.003 |
| s10 | 0.265 | 0.134 | 0.090 | **0.51** | 0.001 / 0.002 |

- WL76 FIA measured, on training sources: native about 0.32, fine16 0.19, coarse 0.13–0.14, contrast 0.69–0.75.
- **Consequence (attribution, non-evidence).** A large part, plausibly most, of WL76's INSUFFICIENT FIT is produced by the deployable q = 1024 single-scatter quadrature noise. That noise is independent per incoming bin and creates spurious incoming contrast. The learned post-collision rule is a smaller part. The design flaw is numerical: a per-geometry Monte Carlo quadrature inside the operator. The first-interaction factorization is not what fails.
- **Not a free fix.** Raising q costs ∝ q (the neural specialist F9 estimates 27–150 × the 75 s rebuild). A deterministic or low-variance SS / W would be a different design. Testing it is a new hypothesis, so it is outside this batch.

### C3. Translate identity behind the WL75 boundary-far failure (objections A1 / B1; neural specialist F1–F2)

**Method.** WL74 Stage 4 source s101 (port 224) is a rigid translate, after a point-group image, of training source s50 (port 3439). The check compares sorted per-incoming-bin collided energy and the tangential radial energy profile.

| Quantity | Value |
|---|---|
| Sorted per-bin energy, relative difference | 8.6e-5 |
| Sorted per-bin energy, replica SE | 8.8e-5 |
| Radial profile, relative difference | 1.9e-4 |

- The responses are identical within noise. Yet A75 scored con_coarse 0.131 on the training sources and **0.73** on s101 (WL75).
- **Consequence.** The W+UV boundary-far failure in WL75 includes failure on physically identical translates. This is consistent with the verified absolute-position inputs, which act as source-location keys:
  - xyz0 / 0.05, xyz / 0.05;
  - face id / 5;
  - the global 11 × 11 dent grid;
  - world-axis pair offsets;
  - a global softmax U.
- The WL76 Gate 4 population excludes point-group images but not translates. Its far strata therefore test both translate and out-of-orbit transfer.

## 3. Why these findings do not change the selection

C1–C3 strengthen objection A6 (UPHELD) and the B1 explanation. They show that two historical failure attributions were confounded:
- WL76 / WL77: SS-quadrature noise;
- W+UV: absolute-position keys.

Neither finding makes H1, H2 or H3 viable. H1's translate advantage is shared by any de-identified control. H2's and H3's decisive objections are unrelated to SS noise.

The obvious follow-ups would each be a **different hypothesis**, which the brief forbids within this batch ("if the hypothesis itself must change, stop"):
- FIA with a deterministic or low-variance SS;
- a de-identified W+UV.

The de-identified W+UV would also not be a novel mechanism. They are recorded below as open questions, not as recommendations.

## 4. Open questions (factual; for the user's decision)

1. How much of WL76 FIA's training-fit failure remains once the deployable SS / W terms are low-noise or deterministic? And at what rebuild cost?
2. Does a W+UV without absolute-position, face-id, global-grid and world-axis inputs (the "W+UV-Q" control) transfer to the translate and orbit sources (BND4_FAR s101 / s105 / s107 type)? And what does it do on the out-of-orbit contexts?
3. Is a classical solver plus closure (the H3 class) admissible as a "learned intrinsic K" under the FNTO contract? This is a contract decision.
4. Do the WL76 / WL77 frozen verdicts need a re-run with corrected statistics? The original records stand either way.

The WL77A Gate 4 references (Part 0, the WL76 population) remain **unscored**.
