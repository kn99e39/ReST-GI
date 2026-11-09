# C1 computational feasibility (Part E)

> WL80B record (workflow `wf_9bbd762a-ae2`, 12 agent runs). Agent texts reproduced verbatim; headings added by the Coordinator. Design-only: nothing implemented, executed, trained or measured. The original C1 architect was not resumable; a stand-in reconstructed C1 from the sealed WL80A records (immutable inputs, sha256 in `inputs.json`).

# WL80B Part E: computational feasibility of C1 / CP-K

Design-only, read-only. Nothing was run, measured or written.

**Labels.**

| Tag | Meaning |
|---|---|
| [EP] | Established physics |
| [PA] | Prior art |
| [HM] | Historical measurement (file:line) |
| [WD] | WL80A design claim |
| [ND] | New derivation (mine) |
| [UH] | Untested hypothesis |

**Short names for cited files.**

| Short name | File |
|---|---|
| XR | `docs/plans/wl80a/cross_review.md` |
| CR | `docs/plans/wl80a/trackB_candidate_records.md` |
| AR | `docs/plans/wl80a/trackB_adversarial_review.md` |
| CD | `docs/plans/wl80a/coordinator_disposition.md` |

**Units and conventions.**
- Every C1 time below is an estimate from operation and byte counts. None is a measurement.
- The hardware reference is the RTX 3080 Ti 12 GB: about 912 GB/s and about 34 TFLOPS fp32 peak [PA: vendor spec, from memory].
- "Calibrated" bandwidth is about 0.63 TB/s. [ND] I derived it from the measured FIA apply: 1 light in 3.9 ms (`76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md` l.255). The bytes that apply reads, per `experiments/fnto_fia76/fia76.py` l.181–190, are SS 448 MB × 3 channels + W 37 MB × 3 + local R 855 MB + tail 133 MB ≈ 2.44 GB (`76_*` l.253). Ballistic B is not counted, so 0.63 TB/s is a lower bound on that apply's efficiency.
- CSR byte cost is 12 B/nnz in this codebase's convention (fp32 value + int64 column index, `fia76.py` l.179). An int32 index gives 8 B/nnz.

## E0. Symbols and the reconstruction check

| Symbol | Meaning | Value |
|---|---|---|
| P, A, O, C | ports, incoming bins, outgoing bins, channels | 3840, 16, 64, 3 |
| N_in | beams (port × incoming bin) | 61,440 |
| N_out | outputs (port × outgoing bin) | 245,760 |
| N_c | graded cells | 2–3e5 [WD AR l.338–353] |
| nnz(W) | W near field within about 3 mfp, about 300 partners per cell | about 1e8 |
| n_it | inner iterations per solve | — |
| b | right-hand sides, = 3 × lights (W is channel-free, so RGB rides as 3 RHS) | — |

**Correction to the Part A reconstruction (ambiguity 6) [ND].**
- If first-collision points are per beam (about 1e3 per beam, AR l.408), storing W1 at point level would need about 6.1e7 points × 300 partners ≈ 1.8e10 nnz, about 150–220 GB. That is impossible.
- The point layer must be composed at build time into beam-level operators:
  - G1 = W1·F_in, beam → cell;
  - S1 = Φ_pt·F_in, beam → (port, bin).
- This is algebraically identical because everything is linear, so points never exist at runtime.
- If points are instead a shared global set (about 1e6, XR l.945): over the 1e5 mm³ body that is about 10 points per mm³, about 0.46 mm spacing. That is just a finer mesh, and "point-exact" no longer holds below that spacing.
- The choice changes construction work by about 60× (E5).

**Runtime operator [ND, from SC l.89–92]:**

```
out = B u + ω_c S1 u + ω_c Φout ψ
ψ   = (I − ω_c W)⁻¹ ω_c G1 u
```

## E1. Persistent storage: resolvent vs factors vs explicit K

| Form | Size (fp32) | Fits 11 GB? |
|---|---|---|
| Explicit dense K, 61,440 × 245,760 × 3 | 4.53e10 floats = **181 GB** (91 GB fp16) [ND] | No |
| Explicit K truncated to 128 nearest receivers × 64 × 3 per beam | 1.51e9 floats = 6.0 GB, plus a coarse tail (240 cells × 16 groups × 3) of 2.8 GB = **≈ 8.9 GB** [ND]. Note: 128 NN alone drops 5.5 % of interior red energy (brief §3a), hence the tail | Marginal, alone |
| Same at rest, stored per D4h orbit (÷16) | ≈ 0.55 GB [ND] | Yes |
| Dense resolvent (I − ωW)⁻¹ | N_c² = 6.25e10 per channel → **250 GB per channel**, about 750 GB for RGB. ω-dependent, so not shared across channels [ND] | No |
| Truncated resolvent: red slab-mode length ≈ 3.6 mm (CD MEDIUM-5), radius about 15 mm | About 450 lateral columns × 22 layers ≈ 1e4 partners per cell → 2.5e9 per channel → about 30 GB RGB [ND/UH] | No |
| **Factors, general/dent geometry**: W near 0.8–1.2 GB; G1 (300–1,000 cells per beam) 0.15–0.7 GB; S1 (13–25 ports × 64 bins per beam) 0.4–1.2 GB; Φout (1e8–3e8 nnz) 1–3.6 GB; far field (split-kernel grid, about 6.4e6 padded points) about 0.1 GB; vectors 3.6 MB per lighting state | **≈ 2.5–6.8 GB** [ND; XR l.924 gives 3–6 GB] | Yes at level L0 |
| Factors at rest (per-face translation templates, AR l.328, l.438) | < 1 GB (W near field still explicit unless Toeplitz/FFT is used) [ND/UH] | Yes |

**Comparison with measured states** [HM]:
- A75 state 29.5 M floats ≈ 118 MB; FIA 1.47 GB (`76_*` l.253).
- A75 rank-16 peak memory above geometry 449 MB (`75_WORKLOG_RANK16_LEARNED_ATO_TRANSFER.md` l.223).
- C1's general-geometry factor state is about 20–60× A75's state.
- It is borderline against the WL76 computational sub-gate "state ≤ 4 GiB" (`76_*` l.127).

**Refinement does not fit the explicit form [ND].**
- nnz(W) scales as cells × partners.
- Halving lateral spacing only (the chair's L1): ×4 × ×4 = ×16 → 1.6e9 nnz, 13–19 GB.
- Halving every graded size (AR l.361): ×64 → about 6.4e9 nnz, 51–77 GB.
- So every level above L0 is "affordability-limited" (CD HIGH-1) unless one of these is used:
  - the explicit near-field radius is held at a fixed number of cells, with the rest moved to a grid far field;
  - W is evaluated matrix-free (compute-bound, about 1e9–1e11 kernel evaluations per apply [UH]).
- The strata that most need refinement are sides, edges, corners and the dent. These are exactly where the lateral FFT/Toeplitz structure is lost. So the decisive BND_FAR refinement study is also the one that does not fit memory.

**Far-field positivity [ND].**
- A precorrected FFT ([PA] Phillips & White, IEEE TCAD 16(10), 1997) subtracts the grid-approximated near interaction. That puts negative entries into the assembled W.
- This conflicts with the passivity condition "W ≥ 0 entrywise, 1ᵀW ≤ 1" (CD MEDIUM-6).
- A non-negative alternative is an Ewald/P3M-type partition-of-unity kernel split: k = s·k + (1−s)·k, with 0 ≤ s ≤ 1, a mass-preserving cloud-in-cell projection and interpolation, and a non-negative FFT convolution. This is non-negative up to FFT rounding [PA: kernel splitting in P3M, from memory; applicability here UH].
- Total far-field mass still has to be renormalised against the exact complement. That remains the main engineering risk, as XR 1.2(d) says.

## E2. Numerical convergence

**Spectral radii [ND]** (diffusion estimate, consistent with AR l.218).
- Extrapolated thickness 11.42 mm; B² = (π/11.42)² + 2(π/101.42)² = 0.0776 mm⁻².
- λ_max(W) ≈ 1/(1 + B²/3) = 0.975.

| Channel | ρ = ω·λ | Gain 1/(1−ρ) |
|---|---|---|
| R | 0.974 | 38 |
| G | 0.965 | 29 |
| B | 0.877 | 8.2 |

Flat-cell diffusion inflation of about 9–18 % in length (CD LOW-11) moves ρ by about 0.1–0.4 % [ND].

**Iterations to relative tolerance ε (per solve; red binds because W is shared) [ND]:**

| Method | R | G | B | Linearity / non-negativity |
|---|---|---|---|---|
| Plain source (Neumann) iteration, n = ln ε / ln ρ | 524 (ε = 1e-6), 262 (ε = 1e-3) | 389 / 194 | 106 / 53 | Exactly linear for a fixed n; monotone ≥ 0; under-estimates (safe for passivity) |
| CG / fixed-degree Chebyshev, ≈ ½√κ·ln(2/ε) | 45 (1e-6), 24 (1e-3) | 39 / 20 | 21 / 11 | Chebyshev with fixed bounds is an RHS-independent polynomial, so exactly linear, but its alternating coefficients only give ≥ 0 to tolerance. CG/GMRES depend on the RHS, so they are linear only to tolerance. CG needs exact volume-weighted reciprocity, which the amended design gave up (AR l.304). So it is GMRES/BiCGStab, or Chebyshev with eigenvalue bounds that are unverified for a non-normal W [ND/UH] |
| DSA / diffusion-preconditioned | ≈ 9–10 sweeps to 1e-6, if ρ_DSA ≈ 0.225·ω | same | fewer | Linear to tolerance |

**Source of the DSA figure [PA]** (search-verified abstract; Adams & Larsen 2002 not checked).
- The continuous-case S_N bound is 0.22466 (Khalil 1988, abstract).
- Transfer to a collision-probability discretisation on a graded mesh with 0.05 × 1.25 mfp cells (aspect ratio 25) is [UH].
- DSA is known to lose effectiveness in multi-D problems with discontinuities. Krylov with a DSA preconditioner is the documented remedy ([PA] Warsa, Wareing & Morel, NSE 147(3), 2004, verified).
- Each DSA iteration adds a diffusion solve on about 2.5e5 cells. Its cost is unmeasured [UH].

**Strongly absorbing media [EP/ND].**
- Source iteration converges like ω. ω = 0.9 needs 106 plain sweeps; ω = 0.5 needs about 20. Acceleration is then unnecessary.
- The cost problem is specific to red and green high albedo. Iteration counts grow like 1/(1−ρ) for plain iteration, √(1/(1−ρ)) for Krylov, and stay about flat for DSA.

**fp32 floor [ND].**
- The residual q − (I − ωW)ψ cancels down to about ψ/38 for red, so the attainable relative residual in fp32 is about eps·38 ≈ 2e-6.
- "Converged to 1e-6" (SC l.115) therefore needs fp64 or mixed-precision accumulation of the residual.
- Passivity has slack of at least about 1e-3·d (XR 1.2(e)), so rounding cannot trip the falsifier, but the stopping test can stall.

**Joint K∘C solve.**
- A coupled iteration inherits ρ ≥ 0.974 (AR l.405, conceded).
- No coupled acceleration is designed [UH].

## E3. Single illumination application (1 lighting state, RGB = 3 RHS)

**Byte counts per apply [ND]:**
- W sweep: 0.8–1.2 GB → 0.9–1.3 ms at peak, 1.3–1.9 ms calibrated.
- Factors (G1 + S1 + Φout, read once): 1.6–5.5 GB → 2.5–8.7 ms calibrated.

| Solver | C1 estimate [ND/UH] | vs FIA measured 3.9 ms (3080 Ti) |
|---|---|---|
| Plain, 524 sweeps | ≈ 0.7–1.0 s | 180–260× |
| Krylov, 45 sweeps | ≈ 65–95 ms | 17–24× |
| DSA, about 10 sweeps | ≈ 16–28 ms + 10 diffusion solves | 4–7× + diffusion |
| Rest, Toeplitz/FFT W (lateral FFT over about 22 layers; 22² layer-pair kernels × 25,600 frequencies) | Sweep likely launch-bound, well under 1 ms; apply set by factor reads | [UH] |

**Anchor correction [HM; overrides part of CD MEDIUM-7].**
- CD says A75's single-light cost is amortised, not measured. That holds only for the WL76 table.
- WL75 measured A75 single-light directly: 1.07 / 1.37 ms, rank 8 / 16, RTX 5080 (`75_*` l.222).
- WL78 reports a measured 3080 Ti apply1 ratio Q78/A75 = 1.01 (`78_WORKLOG_SOURCE_IDENTITY_LEAKAGE.md` l.339), so A75 1-light was measured on the 3080 Ti too. Its absolute value is not in the worklog text.
- XR's "50–150× A75" (l.927) sets a 3080 Ti estimate against a 5080 measurement. That is a cross-hardware comparison.

## E4. Many applications

### 64 lights (b = 192)

**Per W sweep [ND]:**
- Bandwidth floor: matrix 0.8–1.2 GB + 2 × 192 MB of vectors → about 1.3–1.8 ms.
- Compute: 1e8 × 192 × 2 = 3.8e10 FLOP. XR (l.929) estimates 8–20 ms; I take 4–20 ms [UH].
- Factor SpMM: 6.5e10–1.8e11 FLOP, about 7–35 ms [UH].

| Solver | C1 estimate | Measured anchors (3080 Ti) |
|---|---|---|
| Krylov, 45 sweeps | 0.19–0.94 s | FIA 0.349 s; A75 0.086 s (`76_*` l.254). C1 = 0.5–2.7× FIA, 2.2–11× A75 |
| DSA, 10 sweeps | 0.05–0.24 s + diffusion | — |
| Plain, 524 sweeps | 2.1–10.5 s | — |

- The WL76 sub-gate "64-light apply ≤ 10 × A75 = 0.86 s" (`76_*` l.127): DSA and low-end Krylov would pass; plain iteration fails [ND].
- The WL78 gate of 2 × A75 (`78_*` l.171) fails under every solver [ND].

**Batching evidence in this codebase [HM-derived].**
- FIA: 64-light 0.349 s = 5.45 ms per light, *worse* than its 3.9 ms single light (`76_*` l.254–255).
- A75 on the 5080: 63.0 ms / 64 = 0.98 ms per light against 1.37 ms single (`75_*` l.221–222), a 1.4× gain.
- The 64-light C1 figures therefore assume an SpMM kernel more efficient than torch CSR has delivered here [UH].

### Full-boundary GI iterations

| Scene | Cost structure | Estimate |
|---|---|---|
| Isolated convex rest slab | No C re-entry, so GI is one K application per lighting state (XR 1.2(f)) [WD/ND] | E3 / E4 figures |
| Dent cavity | Small re-entry loop | [UH] |
| Multi-domain, nested (cold inner solve per outer iteration) | n_out × T_solve. WL70 fixture: 51 outer iterations, CAK 0.085 s (XR l.912, not re-verified) | 51 × (17–95 ms) ≈ 0.9–4.8 s per GI solve |
| Multi-domain, warm-started or joint | XR estimates 0.3–0.5 s, AR 1–10 s | [UH] |

The path-traced re-render, comparison class 3, takes 1.20–2.40 s on a different, camera-image workload (XR l.914–915).

## E5. Initial construction (counts, not times)

**Measured anchor [HM].**
- The FIA sampled analogue: 61,440 beams × q = 1,024 ≈ 6.3e7 paths, each with 2 ray–mesh intersections and a CPU KD-tree query (`fia76.py` l.65–89). It took 74.3 s, 75.0 s in total (`76_*` l.252); a local rest build took 49 s (l.145).
- q = 1,024 left single-scatter noise at 0.45–0.48 of the M norm. The noise floor is approached only at q ≥ 65,536 (`77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md` l.77–83).
- Linear scaling would put a sampled C1 at about 80 min (XR 1.2(b), inference).

**Deterministic route [ND]:**

| Factor | Work |
|---|---|
| G1 / S1 / Φout, 1e6 global points × 9–25 ports × 64 bins × 64 nodes | ≈ 1e10–1e11 exponential/bin evaluations |
| Same with per-beam points (1e3 per beam) | ≈ 6e12, about 60× more |
| W by ray telescoping: 2.5e5 cells × 1e3–1e4 rays × 30–60 graded-cell segments | ≈ 7.5e9–1.5e11 analytic segment deposits |

- Directional quadrature adds ray-effect error that must be refinement-checked, as in MOC [PA, from memory].
- At rest, translation templates reduce the work to the boundary contexts: about a few thousand template beams (my count: about 576 near-edge top ports, ÷8 by in-plane symmetry, plus side contexts, × 16 bins). That is about 3–8 % of the full-geometry work [ND].
- No kernel exists; the timing is [UH]. XR's "sub-second to a few seconds" and AR's "1–5 s" rest on that unbuilt kernel (CD MEDIUM-7; XR 1.2(b)).

## E6. Geometry rebuild and local update

| Case | Work |
|---|---|
| Rigid motion | No factor rebuild (body frame) [WD]. B and the shared geometry preparation still apply |
| Albedo edit | Factors are free, but the solve repeats, and any tabulated K must be fully retabulated (ω enters the resolvent) [ND] |

**Dent, on-the-fly [ND].**
- The vertical stretch map (AR l.322) deforms every depth layer in the footprint columns. The dirty cells are therefore full-depth columns: π·25²/1.5625 ≈ 1,257 columns × 22 layers ≈ 2.8e4 cells, plus W partners within 3–10 mfp. That is about 5e4–1e5 of the W rows, about 20–35 % of the cells.
- Dirty beams number 1,000–1,600 ports (XR l.974): 16k–26k beams, **26–42 % of all beams**.
- "Local" therefore means about a third of the full construction work. After it, a global solve absorbs D_T ≠ D_G with no dirty-set oracle [WD].

**Comparators.**

| Item | Cost | Source |
|---|---|---|
| A75 rebuild | 0.015 s | `76_*` l.252 |
| Q78 model-specific rebuild | 0.102 s; shared inputs 0.085 s | `78_*` l.343 |
| WL59 selective update | 0.33 s, of which geometry + B is 0.32 s | `59_WORKLOG_PERSISTENT_INTRINSIC_FS.md` l.182 |
| FIA rebuild | 75 s | `76_*` l.252 |
| WL56 oracle, two-medium dent (different fixture) | 371.40 s full; 188.93 s selective; TR 175.50 s | [HM; AGENTS.md; WL56 l.5, 132–133 — WL56 is outside my listed read set and was read only via search] |

- Any C1 update in the WL59-style pipeline inherits the about 0.32 s geometry/B floor [ND].
- The C1 dent rebuild is a counted estimate only.

## E7. MC and classical solvers

**MC reference [HM].** A packet is one (geometry, port, bin, replica) beam: seeds `…+slot·1e4+bin·100+rep` (`71_WORKLOG_SPATIAL_MAP_COVERAGE_GENERALIZATION.md` l.322).

| Run | Per-packet time | Source |
|---|---|---|
| WL71 preflight | 3.37 s | l.147 |
| WL71 production (2.92 h / 3,072) | 3.42 s | l.155 |
| WL72 | 3.02 s | l.68 |
| WL73 | 2.39 / 2.82 s | l.127–128 |
| WL75 (RTX 5080) | 2.12 s | l.153 |

- A full-K MC tabulation of one 32M-photon replica takes 61,440 × 3.4 s ≈ 57 GPU-h [ND; CD's "≈ 60 GPU-h" is the same derivation].
- C1 tabulation by batched solves is estimated in E8 at minutes. That is deterministic discretisation bias against MC noise, a different error type [UH].

**Classical solvers:**

| Solver | Cost and fit |
|---|---|
| **Discrete ordinates / MOC** [PA, from memory] | Matrix-free sweeps (cells × directions, or tracks × segments) need no stored W. That removes the 0.8–1.2 GB W and the refinement wall in E1. Same ρ, same need for DSA/CMFD. Angular discretisation inside the medium (ray effects) is mitigated by the first-collision source, which is exactly C1's split. C1's ray-telescoped W is in effect a precomputed MOC sweep. Trade: memory for per-sweep time [ND]. GPU sweep ordering is a known difficulty [PA] |
| **Collision probability** (C1 itself) [PA] | Dense CP matrices are classically limited to small regions; C1 needs truncation plus a far field |
| **Diffusion** [EP] | Milliseconds and sparse, but invalid within about 1–2 mfp of boundaries and sources. Gives only a P1 exit angle, so it cannot carry 64-bin grazing structure. Weak for blue (ω = 0.9). Usable only as the DSA preconditioner |
| **Per-state MC** | About 2–3.4 s for a 32M-photon state (HM per-packet cost). Noise level for a full-boundary state is unverified. C1 per state is estimated 20–130× cheaper and noise-free [UH] |

## E8. Caching vs solving per illumination: break-even

**Model [ND].** At fixed geometry:
- on-the-fly cost: T_ofl(n) = n·t_s;
- cached cost: T_cache(n) = T_tab + n·t_K.

Tabulation is itself N_cols batched solves (cost per column ≈ t_s; RHS sparsity barely matters), so:

**N\* = T_tab / (t_s − t_K) ≈ N_cols · t_s / (t_s − t_K) ≈ N_cols when t_K ≪ t_s.**

Representative t_s and t_K (64-light batches):
- t_s ≈ 0.8–15 ms per light (DSA to Krylov);
- t_K ≈ 0.2–0.6 ms per light for a truncated 8.9 GB K, as about 2e11 FLOP of block-GEMM per batch;
- single-light t_K ≈ 10–15 ms (bandwidth-bound read of 8.9 GB).

| Regime | N_cols | Tabulation estimate | Break-even, lighting states | Break-even, in GI terms |
|---|---|---|---|---|
| General/dent geometry, full | 61,440 | 960 batches × 0.05–0.94 s ≈ 1–15 min | ≈ 6e4 | ≈ 1,200 nested GI solves at 51 outer iterations |
| Rest, exact D4h | ≈ 3,840 (more for on-plane orbits) | ≈ 4–56 s | ≈ 4e3 | ≈ 75 nested GI solves, or 60 batches of 64 lights |
| Dent, dirty-column retabulation | 16k–26k | ≈ 0.3–6 min | ≈ 2e4 | ≈ 300–500 GI solves per dent state |

**Verdict [ND/UH].**
- **Static geometry, many lights or GI iterations:** caching wins once roughly N_cols lighting states have been applied. That is a few thousand at rest (if D4h orbit storage is accepted) and about 6e4 otherwise.
- **The cached explicit K is a lossy, truncated table.** It needs about 8.9 GB in general and cannot be resident together with the 2.5–6.8 GB of factors on 12 GB. It must be streamed to the host (PCIe, about 0.3–0.4 s per copy [UH]) or tabulated in compressed form.
- **Dirty-column retabulation is itself approximate.** The resolvent is global, and red's tail extends beyond 128 NN.
- **Per-frame deformation:** tabulation never amortises. On-the-fly C1 is the only exact route, at about 17–95 ms per lighting state single-light (4–24× FIA's measured 3.9 ms) or 0.05–0.94 s per 64 lights. That is comparable to, or above, the about 1–2 s class-3 re-render (different workload) once multi-domain GI loops multiply it.
- **Storing the resolvent is never viable** (E1).

## E9. Baseline terminology

- C1 has no learned representation, so "Neural Full Recompute" does not apply to C1 itself.
- Its internal comparator is full versus local factor rebuild (F/S).
- Cycles/MC re-rendering is comparison class 3 (FD l.199; `BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` gate).

## E10. Summary

- **Feasible at L0 on 12 GB (estimate):** factors about 2.5–6.8 GB; rest under 1 GB with templates.
- **Not feasible:**
  - any refinement level above L0 in the explicit form (13–77 GB);
  - an explicit dense K (181 GB);
  - a stored resolvent (250 GB per channel).
- **Iteration cost is decisive.** Red needs 524 plain sweeps against about 45 Krylov or about 10 DSA sweeps. DSA robustness on 25:1 graded cells is unverified, and the coupled K∘C acceleration is undesigned.
- **The pFFT far field conflicts with W ≥ 0.** A non-negative kernel split is a possible fix [UH].
- **Measured anchors favour the learned operators by about 4–260×** per single application and about 2–120× per 64-light batch, depending on the solver.
- **C1's rebuild speed rests on an unbuilt deterministic quadrature.** The sampled analogue took 74.3 s and was noise-limited.
- **No C1 cost has been measured.**

Sources:
- [Warsa, Wareing & Morel, NSE 2004 (ANS)](https://ans.org/pubs/journals/nse/article-2431)
- [OSTI record, Krylov methods with material discontinuities](https://www.osti.gov/biblio/976408)
- [Khalil, consistently formulated DSA spectral radius (OSTI)](https://www.osti.gov/biblio/7186043)
- [Larsen DSA report (LANL)](https://sgp.fas.org/othergov/doe/lanl/lib-www/la-pubs/00248634.pdf)
- [Phillips & White, precorrected-FFT (DTIC)](https://apps.dtic.mil/sti/pdfs/ADA334811.pdf)
- [Phillips & White, precorrected-FFT (Scilit)](https://scilit.com/publications/e54e81859078e95a632970db104b55b0)
