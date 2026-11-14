# WL95 Stage B: representation evidence review with a matched-norm offline calculation

I edited no repository file. I trained nothing, used no GPU and touched no checkpoint or result. On LabServer63 (CPU only, `CUDA_VISIBLE_DEVICES=""`) I wrote only under `~/wl95_calc/`:
- the three scripts `wl95_calc.py`, `wl95_calc2.py`, `wl95_calc3.py`;
- the results `wl95_calc.json`, `wl95_calc2.json`, `wl95_calc3.json`, `S1_ops.npz`;
- the logs `run*.log` and `__pycache__`.

Local copies of the scripts are `C:\Users\dna10\AppData\Local\Temp\claude\462fa2b2dbb71cdd\46d65e15-e083-4057-875f-9282fb68aa5e\scratchpad\wl95_calc.py`, `...\wl95_calc2.py` and `...\wl95_calc3.py`. Their md5 matches the server copies.

**Main result.**
- [MEAS, this calculation] In the exact WL94D operator metric, the all-zero operator scores **1.000**. The WL94D learned S1 operators score 0.961–1.044, which is the zero-operator level.
- In the same L1 norm:
  - the Frobenius-optimal rank-16 SVD scores 0.891 (it has signed entries);
  - WL92's non-negative NMF16 scores 0.603;
  - an L1-targeted non-negative rank-16 fit scores **0.380**.
- On the actual held-out panel incident fields (WL94D field metric), non-negative rank-16 operators reach a field rel-L1 of **0.36–0.54** at peak 0.35–0.44. The learned operators are at 0.846–0.892, with peak 0.22–0.33.
- Within the factorizations tried, rank 16 caps peak retention near 0.44–0.49 on the field metric.

---

## Part 1 — Inventory of representability numbers and their norms

### 1a. WL92 (`experiments/ecro_attr92/diag92.py`, results `results/ecro_attr92/results/local.json`)

| Quantity | Exact definition (code) | Values |
|---|---|---|
| Flux-transfer matrix | X[bi,bo] = M64[bi,bo]·zb_bi·zb_bo, with zb = ((b//16)+0.5)/4 (`diag92.py:114-119`, `:160`) [FACT] | — |
| EY16 `eckart_young_rank16_fro` | sqrt(Σ_{i>16}σ_i² / Σσ_i²) of the SVD of X, **per colour**, Frobenius (`diag92.py:161-168`) [FACT] | S1 0.520/0.458/0.461; B 0.030/0.028/0.040; S2 0.204/0.335/0.234 [MEAS, WL92:137-139]. **Reproduced exactly here** (svd16 fro). |
| Rank for 0.9/0.99/0.999 energy | cumulative σ² of X (`:169`) [FACT] | S1 25–33 / 46–59 / 61–64 [MEAS] |
| `mat_metrics` `fro` / `l1` / `peak` / `mass` | All on X, per colour (`diag92.py:122-130`) [FACT]:<br>• fro = ‖X̂−X‖_F/‖X‖_F<br>• l1 = Σ\|X̂−X\|/ΣX<br>• peak = Σ_bi X̂[bi, argmax_bo X[bi,:]] / Σ_bi X[bi, argmax] (operator-level, per-row true peak, unweighted by any field)<br>• mass = ΣX̂/ΣX | S1 SVD16 l1 0.87–0.93, peak 0.49–0.56.<br>S1 NMF16 fro 0.544/0.485/0.488, l1 0.59–0.65, peak 0.47–0.54, mass 1.02–1.04 [MEAS, local.json] |
| NMF16 | Multiplicative-update Frobenius NMF **of X** (4,000 iterations, 16 restarts), mapped back as Cn = W/zb, An = Hᵀ/zb (`diag92.py:97-111`, `:165-166`) [FACT] | — |
| Learned `a cᵀ` rows | Per (patch, colour) `mat_metrics` against M64 of the patch material, then averaged; WL86 frozen generators on the E2 T1 geometry (`diag92.py:177-195`) [FACT] | Panel fro 0.877/0.885, peak 0.135/0.112, mass 0.76/0.68, l1 1.05–1.07 [MEAS] |
| WL92A OWN `a cᵀ` | Same `mat_metrics` on the E4 T1 geometry (`own92a.py:345-380`) [FACT] | Peak 0.207/0.212, fro 0.80–0.81, mass 0.80–0.84 [MEAS, WL92A:145-151] |
| Learned participation rank | SVD of the **unweighted** Ml, one patch, green channel (`diag92.py:193`) [FACT] | ≈5.5, against a material participation rank on flux X of 37–44. Not matched [FACT]. |
| Unweighted EY16 | Plain M64 Frobenius (`diag92.py:173`, "not used") [FACT] | Panel 0.19–0.37. Not comparable [MEAS] |

### 1b. WL94D (`experiments/ecro_s94d/s94.py:sender`, results `results/ecro_s94d/sender/SN_*.json`)

| Quantity | Exact definition (code) | Learned values (train / val / CK2) |
|---|---|---|
| `operator_rel_l1` | A = Σ_r c·a on the 42 S1 patches; Σ\|A−M1\|·zb_in·zb_out / Σ M1·zb_in·zb_out. Pooled over (patch, in, out, c) and over the items of the split, split into train-excited and never-excited bins (`s94.py:314-323`; zb at `:288` uses scene86 NZ=4, NPHI=16). This is the **same norm as WL92 `l1`** but pooled over colours and patches [FACT]. | 0.964–1.007 / 0.961–1.002 / 0.987–1.044 [MEAS]. Never-excited bins do not exist (`None`). |
| `rel_l1_flux` | Bl = a(cᵀs), Bm = s·M64[mat], panel only; Σ\|Bl−Bm\|·W / Σ Bm·W, pooled over (lighting, patch, out-bin, c, items) (`s94.py:297-300`, `:287`) [FACT] | val 0.846–0.892; CK2 1.075–1.629 [MEAS] |
| `energy_ratio` | Σ Bl·W / Σ Bm·W (`:300`) [FACT] | val 0.828–1.090; CK2 0.557–1.749 [MEAS] |
| `peak_ratio` | pk = argmax_d Bm per (l, n, c); Σ Bl[pk] / Σ Bm[pk] (`:301-302`). This is **field-level**, at the field's response peak [FACT]. | val 0.224–0.327; CK2 0.038–0.258 [MEAS] |
| `delivery_*` | Panel-only Bl/Bm flown with the fixed `rw`, then the exact receiver M64; exitance with W (`:303-312`); box = patches 1130:1238 hard-coded [FACT] | e.g. val delivery_ratio 0.79–1.09 [MEAS] |
| Teacher loss | mean(((Bl−Bm)/den)²), den = out-bin mean + 0.05 × (patch, bin, ch) mean (`s94.py:57-60`) [FACT] | T: 6.42→4.80 / 6.29→5.57 [MEAS, 94D:92-93] |
| CK2 flight floor | Exact S1 sender + exact receiver vs e2_p, exitance with W (`s94.py:152-162`) [FACT] | rel L1 0.53 (box A) to 0.94 (all receivers) [MEAS] |
| G_PHYS | Exact-sender exitance vs 256-spp label, rel L1 (`s94.py:139-165`) [FACT] | 0.641 [MEAS] |

**[FACT] Defect found in the W weight.**
- `an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, so `W = ((d//8)+0.5)/8·2π/64` (`an94.py:32`).
- The actual bin layout is NZ=4 rings × NPHI=16 sectors, with bin = iz·16 + iφ (`ecro86.json:47`; `pt86.py:153-156`; `scene86.py:15`).
- Effect: inside each ring, W gives sectors 0–7 the cosine (iz+0.25)/4 and sectors 8–15 the cosine (iz+0.75)/4. In the grazing ring that is a 1:3 ratio. The ring sum is still correct.
- So the outgoing weight is azimuth-dependent, which is not physical.
- Where it is used:
  - the WL94D field and delivery metrics (`s94.py:287`);
  - G_PHYS and the CK2 flight floor (`s94.py:129, 157`);
  - WL94 exitance `an94.ex` (`an94.py:37, 104`; amplitudes and slopes);
  - WL94B (`b94.py:47`).
- T_ENERGY and the operator metric use the correct zb (`s94.py:179-181`, `:288`).
- [MEAS, this calculation] M64 S1 maximum directional albedo is 0.825 with the correct weight (it reproduces T_ENERGY 0.825) and 0.899 with W94.
- [MEAS] On the bound arms below, W94 vs the correct weight moves field rel-L1 by ≤ 0.02 and energy by ≤ 0.04. The effect on the learned WL94D values and on the WL94 exitance amplitudes is unmeasured [INF: probably small, sign unknown].
- The task text's own formula `zb=((arange(64)//8)+0.5)/8` copies this defect. I computed both W94 and the true weight WT, and used the correct ring zb for the operator metric, as s94 does.

### 1c. Pairs that are NOT comparable

1. **WL94D statement at 94D:109/147/178**: learned operator ≈1.0 versus "rank-16 Eckart–Young floor 0.46–0.52". This compares flux-L1 (pooled) with flux-Frobenius per colour. It is not matched.
   - [MEAS here] In the WL94D metric the zero operator is exactly 1.000.
   - The SVD16 that attains the 0.46–0.52 Frobenius residual scores 0.891 in the L1 metric, and its signed entries make it inadmissible for the non-negative `a cᵀ`.
   - The matched admissible references are NMF16 at 0.603 and the L1-targeted NMF16 at 0.380.
2. **94D:109**: "WL92 keeps 47–54 % of the mirror-bin flux; learned keep 21–33 % of the peak on validation fields". The first is the operator-level per-row peak (`diag92.py:126-129`); the second is the field-level peak at the Bm argmax (`s94.py:301-302`).
   - [MEAS here] For the same NMF16, the operator peak is 0.518 and the val field peak is 0.436. That is a 0.08 gap from the metric alone.
3. **WL92 learned `fro` / `peak` vs WL94D operator metric.** WL92 is per (patch, colour) mean Frobenius on one geometry (E2 T1; WL92A E4 T1). WL94D is pooled L1 over items. The WL94D models' Frobenius, mass and operator peak were never computed [FACT].
4. **WL92B `encq_vs_q` (0.36) and `Bq_vs_q`.** These use unweighted radiance rel L1 against the physical q (`cf92b.py:235-238`). They are not W-weighted and not M64-referenced, so they are not comparable to the WL94D field rel-L1 against Bm.
5. **Learned participation rank (unweighted) vs material rank-for-energy (flux).**
6. **Any field or operator number vs region E_H**: the norms differ, and E_H is reader-weighted and normalised by total radiance.
7. **Different references.** WL94D field metrics use Bm = M64 bin-projected (teacher), not physical q. The CK2 flight floor uses the path-traced e2_p.

### 1d. Matched comparisons needed to judge representation capacity

- **Operator level, same norm and same admissibility for learned and bound:**
  - flux-L1 (WL94D), flux-Frobenius per colour (WL92) and operator peak/mass;
  - learned evaluated per (geometry, patch) for the WL94D models;
  - bounds restricted to non-negative rank 16, both Frobenius-optimal (NMF16) and L1-targeted;
  - plus the zero-operator and rank-1 references.
- **Field level on the actual incident distribution:**
  - the same pooled rel-L1, energy and field-peak on the 96 training and 24 validation panel fields;
  - material-level and position-conditioned bounds, fitted on train and evaluated on val, with the correct W reported next to W94.
- **Peak/lobe retention at matched energy.** L1-optimal fits under-emit (energy 0.6–0.7) and Frobenius-optimal fits are near 1, so peak and L1 must be read jointly with energy. The overlap Σmin(A,M)/ΣM = (1 + mass − L1)/2 for non-negative A is a single combined summary.
- **Energy**: albedo per incident bin with the correct zb, plus the field energy ratio.
- **Not done here (needs the models, i.e. GPU or a CPU model load):**
  - learned WL94D Frobenius, mass and operator peak;
  - sender delivery and E_H for the bound operators substituted at S1;
  - receiver-side use on E1 fields (the tied c is used twice).

---

## Part 2 — Calculation (CPU, LabServer63)

### Inputs
- `~/rest_x92/results/nmf.npz`: Mq [6,64,64,3], Cn/An [3 mat, 3 ch, 64, 16].
- The 96 train and 24 val items from `~/rest_x94/results/select94.json` (`splits/train|val/files`), with `s` from `~/rest_ecro86/data/<split>/<file>`.
- Panel = 1088..1129, asserted pmat == 1.
- B = patches with pmat 0 (256); S2 = pmat 2 (32).
- Rows: S1 16,128 train / 4,032 val; B 98,304 / 24,576; S2 12,288 / 3,072.

### Sanity check
[MEAS] svd16 fro 0.520/0.458/0.461 (S1), 0.030/0.028/0.040 (B), 0.204/0.335/0.234 (S2) equal WL92 `local.json`. M64 S1 albedo max is 0.8247 with the correct weight, equal to WL94D T_ENERGY 0.825.

### Arms
- **svdk**: truncated SVD of flux X per channel, mapped back to A (signed).
- **svd16_clip0**: svd16 clipped at 0.
- **nmf16_wl92**: Cn·Anᵀ.
- **nmf16_l1irls**: IRLS weighted-Frobenius NMF towards min flux-L1, initialised from NMF16 (40×100 iterations).
- **rrr{k}**: field-adapted reduced-rank regression. It minimises Σ‖S(A−M)diag(W)‖_F² over pooled training fields per channel, and is zero on the out-of-span subspace.
- **fa_nmf16_l1**: field-adapted non-negative rank 16; weighted-L1 IRLS NMF with weights E_tr[bi]·W94[bo], where E_tr is the pooled training panel irradiance per incident bin.
- **per-patch**: the same objective per panel patch index (position-conditioned).
- **zero**: A = 0.

### Table 1 — S1 panel (material 1)

Operator columns use the WL94D metric (i). fro is per colour R/G/B. Field columns are metrics (ii)–(iv) with W94, as WL94D coded them.

| Arm | Op L1 (i) | fro R/G/B | Op peak | Op mass | Train field L1 / energy / peak | **Val field L1 / energy / peak** | Val with true W (L1 / energy) |
|---|---|---|---|---|---|---|---|
| zero | **1.000** | 1/1/1 | 0 | 0 | 1.000 / 0 / 0 | 1.000 / 0 / 0 | — |
| svd1 | 1.281 | .939/.931/.931 | .058 | .731 | 1.265 / .851 / .055 | 1.249 / .757 / .044 | 1.230 / .730 |
| svd2 | 1.202 | .889/.874/.874 | .109 | .731 | 1.115 / .814 / .105 | 1.146 / .739 / .083 | 1.139 / .731 |
| svd4 | 1.120 | .797/.770/.770 | .202 | .731 | 1.003 / .814 / .181 | 1.040 / .740 / .154 | 1.029 / .731 |
| svd8 | 1.150 | .680/.637/.639 | .338 | .936 | .959 / .979 / .295 | 1.029 / .986 / .271 | 1.017 / .969 |
| svd16 | 0.891 | .520/.458/.461 | .539 | .937 | .643 / .964 / .467 | .726 / .992 / .483 | .720 / .968 |
| svd24 | 0.707 | .400/.317/.323 | .689 | .937 | .463 / .965 / .574 | .519 / .989 / .619 | .522 / .968 |
| svd32 | 0.548 | .320/.226/.235 | .791 | .988 | .346 / .995 / .668 | .353 / .989 / .706 | .356 / .983 |
| svd48 | 0.076 | .178/.069/.086 | .936 | .989 | .039 / .996 / .832 | .049 / .990 / .847 | .060 / .984 |
| svd64 | 0 | 0 | 1 | 1 | 0 / 1 / 1 | 0 / 1 / 1 | — |
| svd16_clip0 | 0.683 | .503/.439/.442 | .539 | 1.145 | .524 / 1.180 / .468 | .580 / 1.212 / .483 | .578 / 1.181 |
| **nmf16_wl92** | **0.603** | .544/.485/.488 | .518 | 1.038 | .444 / 1.068 / .448 | **.536 / 1.094 / .436** | .536 / 1.071 |
| **nmf16_l1irls** | **0.380** | .603/.549/.551 | .462 | .689 | .319 / .740 / .370 | **.364 / .706 / .347** | .373 / .695 |
| **fa_nmf16_l1** (field-adapted, non-negative) | 0.378 | .602/.549/.551 | .462 | — | .317 / .743 / .370 | **.362 / .709 / .346** | .371 / .697 |
| rrr16 (field-adapted L2, signed) | 1.028 | .572/.492/.495 | .517 | .956 | .682 / .978 / .489 | .802 / .943 / .490 | .821 / .962 (rrr16 fitted with true W) |
| rrr16_clip0 | 0.768 | .550/.466/.470 | .517 | 1.216 | .558 / 1.219 / .490 | .606 / 1.221 / .490 | — |
| rrr24 / rrr32 (L2) | .853 / .683 | — | .658 / .769 | — | .495 / .362 L1 | .614 / .512 L1; peak .589 / .676 | — |
| per-patch rrr16 (L2) | mean 0.775 | — | — | — | — | .583 / .711 / .434 | — |
| **per-patch L1 non-negative rank 16** | mean 0.505 (0.38–0.63) | — | — | — | **.263** / .787 / .443 | **.451 / .592 / .317** | — |

**Field-adapted L1 non-negative ladder**, random initialisation then IRLS. Values are op L1; val field L1 / energy / peak.

| Rank | Op L1 | Val field L1 / energy / peak |
|---|---|---|
| 4 | .798 | .739 / .284 / .125 |
| 8 | .660 | .618 / .416 / .208 |
| 16 | .464 | .406 / .628 / .392 |
| 24 | .346 | .355 / .666 / .478 |
| 32 | .233 | .146 / .877 / .758 |

At rank 16 the random initialisation reaches a worse local optimum than the NMF16-initialised fit (.362).

**WL94D learned, for comparison** [MEAS, `SN_*.json`]:
- Op L1: val 0.961–1.002; CK2 0.987–1.044; train 0.964–1.007.
- Val field L1 / energy / peak: 0.846–0.892 / 0.828–1.090 / 0.224–0.327. Examples:
  - T: s8601 .855 / .992 / .316; s8602 .846 / 1.090 / .327.
  - own: .878 / .828 / .224 and .850 / .932 / .261.

**Incident distribution at the panel** [MEAS, this calculation]:
- The training span is full rank (64 at tolerance 1e-10); 48 dimensions hold 99 % of the energy and 58 hold 99.9 %.
- Median lit incident bins per (light, patch) row: **1**. Median top-1 bin share: **0.94**.
- 35 bins hold 90 % of pooled training irradiance.
- Ring shares of irradiance, grazing → normal:
  - train 0.043 / 0.146 / 0.289 / 0.522;
  - val 0.067 / 0.092 / 0.439 / 0.402.
- The exact field-L1 numerator for NMF16 is 0.79× the single-bin approximation.

### Table 2 — Contrast materials (val field metrics with W94)

| Material | Arm | Op L1 | fro R/G/B | Op peak | Val field L1 / energy / peak |
|---|---|---|---|---|---|
| B floor (mat 0) | svd8 | .099 | .083/.081/.099 | .880 | .065 / 1.000 / .885 |
| | svd16 | .033 | .030/.028/.040 | .967 | .017 / 1.000 / .982 |
| | nmf16_wl92 | .032 | .031/.029/.042 | .965 | .016 / 1.001 / .978 |
| | nmf16_l1irls | .024 | — | .949 | .012 / .998 / .951 |
| | rrr16 | .055 | — | .949 | .012 / 1.000 / .959 |
| S2 sphere (mat 2) | svd16 | .331 | .204/.335/.234 | .687 | .300 / .986 / .463 |
| | nmf16_wl92 | .259 | — | .669 | .225 / 1.025 / .417 |
| | nmf16_l1irls | .199 | — | .602 | .181 / .909 / .306 |
| | rrr16 (L2) | .369 | — | .627 | **.175 / .996 / .742** |
| | rrr16_clip0 | .340 | — | .627 | .164 / 1.010 / .742 |
| | svd32 | .153 | — | .871 | .144 / .999 / .791 |

- Training span: B 55 dimensions (99 % in 36); S2 52 dimensions (99 % in 39).
- S2 caveat: flat-frame M64 and patch-centre-frame binning (`quad87.py:15-16`).

### Code run

`~/wl95_calc/wl95_calc.py` (main; run time 15m52s):
```python
"""WL95 Stage B offline CPU calculation (read-only inputs; writes only ~/wl95_calc/).
Matched-norm representability bounds for the M64 local operator (S1 panel mat 1; contrast B mat 0, S2 mat 2).
Metrics copied from experiments/ecro_s94d/s94.py:sender (operator, rel_l1_flux, energy_ratio, peak_ratio)."""
import json, math, os, sys
import numpy as np

H = os.path.expanduser('~')
OUT = os.path.join(H, 'wl95_calc')
Z = np.load(os.path.join(H, 'rest_x92/results/nmf.npz'))
Mq = Z['Mq'].astype(np.float64)          # [6, bi, bo, c]
Cn, An = Z['Cn'], Z['An']                # [3 mat, 3 c, 64, 16]
SEL = json.load(open(os.path.join(H, 'rest_x94/results/select94.json')))['splits']
R86 = os.path.join(H, 'rest_ecro86')
NB = 64
PANEL = np.arange(1088, 1130)
zb = ((np.arange(NB) // 16) + 0.5) / 4                 # true layout NZ=4 x NPHI=16 (scene86; used by s94.operator and diag92.flux_w)
W94 = ((np.arange(NB) // 8) + 0.5) / 8 * (2 * np.pi / NB)   # an94.W as coded (NPHI=8, NZ=8) -> used by s94 field metrics
WT = zb * (2 * np.pi / NB)                              # physically consistent cos dOmega for the true layout
RANKS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
ZZ = zb[:, None] * zb[None, :]

def op_metric(A, M):                                    # s94.operator: pooled over (in, out, c)
    return float((np.abs(A - M) * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum())

def fro_c(A, M):                                        # diag92.mat_metrics 'fro' per channel (flux X)
    return [float(np.linalg.norm((A[..., c] - M[..., c]) * ZZ) / np.linalg.norm(M[..., c] * ZZ)) for c in range(3)]

def peak_op(A, M):                                      # diag92 'peak' (operator-level, per-row true peak), pooled channels
    X, Xt = A * ZZ[..., None], M * ZZ[..., None]
    pk = Xt.argmax(1)
    ii = np.arange(NB)
    return float(sum(X[ii, pk[:, c], c].sum() for c in range(3)) / sum(Xt[ii, pk[:, c], c].sum() for c in range(3)))

def svd_k(M, k):
    A = np.zeros_like(M)
    for c in range(3):
        U, s, Vt = np.linalg.svd(M[..., c] * ZZ)
        A[..., c] = ((U[:, :k] * s[:k]) @ Vt[:k]) / ZZ
    return A

def nmf_l1(M, Cm, Am, outer=40, inner=100):
    """IRLS weighted-Frobenius NMF towards min L1 of the flux matrix, initialised from WL92 NMF16 (feasible nonneg rank-16)."""
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c] * ZZ
        Wf = Cm[c] * zb[:, None]; Hf = (Am[c] * zb[:, None]).T
        d = 1e-3 * V.mean(); eps = 1e-30
        for _ in range(outer):
            Om = 1.0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(inner):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + eps)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + eps)
        A[..., c] = (Wf @ Hf) / ZZ
    return A

def load_fields(split, pmat_id):
    rows = []
    for rec in SEL[split]['files']:
        z = np.load(os.path.join(R86, 'data', split, rec['file']))
        s = z['s'].astype(np.float64)                   # [L, N, 64, 3]
        sel = PANEL if pmat_id == 1 else np.nonzero(z['pmat'] == pmat_id)[0]
        if pmat_id == 1:
            assert (z['pmat'][PANEL] == 1).all()
        rows.append(s[:, sel].reshape(-1, NB, 3))       # rows (l, n)
    return rows                                         # list per item

def field_metrics(rows, A, M, W):
    acc = dict(l1=0., m=0., l=0., pl=0., pm=0.)
    for S in rows:
        Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
        acc['l1'] += float((np.abs(Bl - Bm) * W[None, :, None]).sum()); acc['m'] += float((Bm * W[None, :, None]).sum())
        acc['l'] += float((Bl * W[None, :, None]).sum())
        pk = Bm.argmax(1)[:, None, :]
        acc['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); acc['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
    return {'field_rel_l1': acc['l1'] / acc['m'], 'energy': acc['l'] / acc['m'], 'peak': acc['pl'] / acc['pm']}

def rrr(rows, M, W, k, rtol=1e-10):
    """Field-adapted rank-k operator: argmin_rank(A)<=k sum ||S (A - M) diag(W)||_F^2 over the pooled training fields, per channel.
    S = U s V^T; Z = best rank-k of diag(s) V^T M D; A = V diag(1/s) Z D^-1 (zero on the out-of-span incident subspace)."""
    A = np.zeros_like(M); info = []
    D = W
    for c in range(3):
        S = np.concatenate([r[..., c] for r in rows], 0)
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        keep = s > rtol * s[0]
        s, Vt = s[keep], Vt[keep]
        P = s[:, None] * Vt
        Y = P @ M[..., c] * D[None, :]
        u, sg, vt = np.linalg.svd(Y, full_matrices=False)
        Zk = (u[:, :k] * sg[:k]) @ vt[:k]
        A[..., c] = (Vt.T @ (Zk / s[:, None])) / D[None, :]
        e = s ** 2 / (s ** 2).sum()
        info.append({'span_rank_tol': int(keep.sum()), 'rank99': int(np.searchsorted(np.cumsum(e), 0.99) + 1),
                     'rank999': int(np.searchsorted(np.cumsum(e), 0.999) + 1)})
    return A, info

def run():
    os.makedirs(OUT, exist_ok=True)
    res = {'note': 'W94 = an94.W as coded (NPHI 8, NZ 8); WT = true-layout cos dOmega; zb = true ring cosine', 'materials': {}}
    for m, name in ((1, 'S1_panel'), (0, 'B_floor'), (2, 'S2_sphere')):
        M = Mq[m]
        tr, va = load_fields('train', m), load_fields('val', m)
        o = {'n_rows_train': int(sum(r.shape[0] for r in tr)), 'n_rows_val': int(sum(r.shape[0] for r in va)),
             'albedo_bin_max_W94': float((M * W94[None, :, None]).sum(1).max()), 'albedo_bin_max_WT': float((M * WT[None, :, None]).sum(1).max())}
        arms = {f'svd{k}': svd_k(M, k) for k in RANKS}
        arms['svd16_clip0'] = np.maximum(arms['svd16'], 0)
        arms['nmf16_wl92'] = np.stack([Cn[m, c] @ An[m, c].T for c in range(3)], -1)
        arms['nmf16_l1irls'] = nmf_l1(M, Cn[m], An[m])
        arms['zero'] = np.zeros_like(M)
        for W, wn in ((W94, 'W94'), (WT, 'WT')):
            for k in (4, 8, 16, 24, 32):
                A, info = rrr(tr, M, W, k)
                arms[f'rrr{k}_{wn}'] = A
                o[f'rrr_info_{wn}'] = info
        A16, _ = rrr(tr, M, W94, 16)
        arms['rrr16_W94_clip0'] = np.maximum(A16, 0)
        tab = {}
        for an, A in arms.items():
            r = {'op_l1': op_metric(A, M), 'fro_c': fro_c(A, M), 'op_peak': peak_op(A, M),
                 'mass': float((A * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()), 'min_entry_rel': float(A.min() / M.max())}
            for W, wn in ((W94, 'W94'), (WT, 'WT')):
                r[f'train_{wn}'] = field_metrics(tr, A, M, W); r[f'val_{wn}'] = field_metrics(va, A, M, W)
            tab[an] = r
        if m == 1:
            pp = {'val_W94': dict(l1=0., m=0., l=0., pl=0., pm=0.)}
            Ap = []
            for j in range(len(PANEL)):
                trj = [r.reshape(-1, len(PANEL), NB, 3)[:, j] for r in tr]
                vaj = [r.reshape(-1, len(PANEL), NB, 3)[:, j] for r in va]
                A, _ = rrr(trj, M, W94, 16)
                fm = field_metrics(vaj, A, M, W94)
                Ap.append((A, vaj))
            acc = dict(l1=0., m=0., l=0., pl=0., pm=0.)
            for A, vaj in Ap:
                for S in vaj:
                    Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
                    acc['l1'] += float((np.abs(Bl - Bm) * W94[None, :, None]).sum()); acc['m'] += float((Bm * W94[None, :, None]).sum())
                    acc['l'] += float((Bl * W94[None, :, None]).sum())
                    pk = Bm.argmax(1)[:, None, :]
                    acc['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); acc['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
            o['rrr16_per_patch_W94_val'] = {'field_rel_l1': acc['l1'] / acc['m'], 'energy': acc['l'] / acc['m'], 'peak': acc['pl'] / acc['pm'],
                                           'op_l1_mean': float(np.mean([op_metric(A, M) for A, _ in Ap]))}
            S = np.concatenate(tr, 0)
            ring = np.array([S[:, 16 * i:16 * i + 16].sum() for i in range(4)]); o['train_ring_share'] = (ring / ring.sum()).tolist()
            o['train_bins_lit_per_row_median'] = float(np.median((S.sum(2) > 0).sum(1)))
            o['train_top1_bin_share_median'] = float(np.median(S.sum(2).max(1) / np.maximum(S.sum(2).sum(1), 1e-30)))
            np.savez_compressed(os.path.join(OUT, 'S1_ops.npz'), **{k: v for k, v in arms.items()})
        o['arms'] = tab
        res['materials'][name] = o
        print('done', name, flush=True)
        json.dump(res, open(os.path.join(OUT, 'wl95_calc.json'), 'w'), indent=1)
    json.dump(res, open(os.path.join(OUT, 'wl95_calc.json'), 'w'), indent=1)

if __name__ == '__main__':
    run()
```

`~/wl95_calc/wl95_calc2.py` (field-adapted non-negative L1 rank 16 and its rank ladder; 58 s):
```python
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/wl95_calc'))
import wl95_calc as C
M = C.Mq[1]
tr, va = C.load_fields('train', 1), C.load_fields('val', 1)
out = {}
for W, wn in ((C.W94, 'W94'),):
    E = sum(r.sum(0) for r in tr)                      # [64 in, 3]
    Ev = sum(r.sum(0) for r in va)
    out['E_share_train_by_ring'] = [float(E[16 * i:16 * i + 16].sum() / E.sum()) for i in range(4)]
    out['E_share_val_by_ring'] = [float(Ev[16 * i:16 * i + 16].sum() / Ev.sum()) for i in range(4)]
    out['bins_holding_90pct_train_E'] = int(np.searchsorted(np.cumsum(np.sort(E.sum(1))[::-1]) / E.sum(), 0.9) + 1)
    A16 = np.stack([C.Cn[1, c] @ C.An[1, c].T for c in range(3)], -1)
    num_exact = sum(float((np.abs(np.einsum('kmc,mdc->kdc', r, A16 - M, optimize=True)) * W[None, :, None]).sum()) for r in tr)
    num_sb = float((E[:, None, :] * np.abs(A16 - M) * W[None, :, None]).sum())
    out['single_bin_approx_ratio_nmf16'] = num_exact / num_sb
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c]
        Om0 = E[:, c][:, None] * W[None, :]
        Wf = C.Cn[1, c].copy(); Hf = C.An[1, c].T.copy()
        d = 1e-3 * V.mean(); eps = 1e-30
        for _ in range(60):
            Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(100):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + eps)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + eps)
        A[..., c] = Wf @ Hf
    out['fa_nmf16_l1'] = {'op_l1': C.op_metric(A, M), 'fro_c': C.fro_c(A, M), 'op_peak': C.peak_op(A, M),
                          'train_W94': C.field_metrics(tr, A, M, C.W94), 'val_W94': C.field_metrics(va, A, M, C.W94),
                          'val_WT': C.field_metrics(va, A, M, C.WT)}
    lad = {}
    for k in (4, 8, 16, 24, 32):
        Ak = np.zeros_like(M)
        for c in range(3):
            V = M[..., c]; Om0 = E[:, c][:, None] * W[None, :]
            rng = np.random.default_rng(9500 + k * 10 + c)
            sc = np.sqrt(V.mean() / k)
            Wf = rng.random((64, k)) * sc + 1e-12; Hf = rng.random((k, 64)) * sc + 1e-12
            for _ in range(300):
                Hf *= (Wf.T @ (Om0 * V)) / (Wf.T @ (Om0 * (Wf @ Hf)) + 1e-30)
                Wf *= ((Om0 * V) @ Hf.T) / ((Om0 * (Wf @ Hf)) @ Hf.T + 1e-30)
            d = 1e-3 * V.mean()
            for _ in range(60):
                Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
                for _ in range(100):
                    Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + 1e-30)
                    Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + 1e-30)
            Ak[..., c] = Wf @ Hf
        lad[k] = {'op_l1': C.op_metric(Ak, M), 'op_peak': C.peak_op(Ak, M),
                  'train_W94': C.field_metrics(tr, Ak, M, C.W94), 'val_W94': C.field_metrics(va, Ak, M, C.W94)}
    out['fa_nmf_l1_ladder_randinit'] = lad
json.dump(out, open(os.path.expanduser('~/wl95_calc/wl95_calc2.json'), 'w'), indent=1, default=float)
```

`~/wl95_calc/wl95_calc3.py` (per-panel-patch position-conditioned non-negative L1 rank 16; 9 min):
```python
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser('~/wl95_calc'))
import wl95_calc as C
M = C.Mq[1]; NP = len(C.PANEL)
tr, va = C.load_fields('train', 1), C.load_fields('val', 1)
trp = [r.reshape(-1, NP, 64, 3) for r in tr]; vap = [r.reshape(-1, NP, 64, 3) for r in va]
acc = {s: dict(l1=0., m=0., l=0., pl=0., pm=0.) for s in ('train', 'val')}
opl = []
for j in range(NP):
    E = sum(r[:, j].sum(0) for r in trp)
    A = np.zeros_like(M)
    for c in range(3):
        V = M[..., c]; Om0 = E[:, c][:, None] * C.W94[None, :] + 1e-6 * E[:, c].max() * C.W94[None, :]
        Wf = C.Cn[1, c].copy(); Hf = C.An[1, c].T.copy(); d = 1e-3 * V.mean()
        for _ in range(30):
            Om = Om0 / np.maximum(np.abs(V - Wf @ Hf), d)
            for _ in range(100):
                Hf *= (Wf.T @ (Om * V)) / (Wf.T @ (Om * (Wf @ Hf)) + 1e-30)
                Wf *= ((Om * V) @ Hf.T) / ((Om * (Wf @ Hf)) @ Hf.T + 1e-30)
        A[..., c] = Wf @ Hf
    opl.append(C.op_metric(A, M))
    for s, rows in (('train', trp), ('val', vap)):
        for r in rows:
            S = r[:, j]
            Bl = np.einsum('kmc,mdc->kdc', S, A, optimize=True); Bm = np.einsum('kmc,mdc->kdc', S, M, optimize=True)
            a = acc[s]; Wd = C.W94[None, :, None]
            a['l1'] += float((np.abs(Bl - Bm) * Wd).sum()); a['m'] += float((Bm * Wd).sum()); a['l'] += float((Bl * Wd).sum())
            pk = Bm.argmax(1)[:, None, :]
            a['pl'] += float(np.take_along_axis(Bl, pk, 1).sum()); a['pm'] += float(np.take_along_axis(Bm, pk, 1).sum())
out = {s: {'field_rel_l1': a['l1'] / a['m'], 'energy': a['l'] / a['m'], 'peak': a['pl'] / a['pm']} for s, a in acc.items()}
out['op_l1_mean'] = float(np.mean(opl)); out['op_l1_range'] = [float(min(opl)), float(max(opl))]
json.dump(out, open(os.path.expanduser('~/wl95_calc/wl95_calc3.json'), 'w'), indent=1)
```

---

## Interpretation

### What is established about rank-16 capacity for S1

All numbers are relative to the M64 teacher Bm, which is the quantity WL94D supervised.

1. **The learned operator sits at the zero-operator level in the WL94D metric.** [MEAS] The zero operator scores 1.000 in the WL94D operator metric; the learned operators score 0.961–1.044.
   - [INF] For non-negative A, overlap Σmin(A,M)/ΣM = (1 + mass − L1)/2.
   - With WL92A OWN mass 0.80–0.84 and L1 ≈ 1, the learned overlap is about 0.4.
   - NMF16 gives 0.72 and the L1-targeted NMF16 gives 0.65.
2. **Matched-norm gap at the operator level.** [MEAS] Admissible non-negative rank 16 reaches ≤ 0.380 (L1-targeted; a local optimum, so the true minimum is ≤ 0.380) and 0.603 (Frobenius NMF16).
   - The learned operators are 0.36–0.66 above these in the same norm.
   - The "0.46–0.52 floor" is not a matched reference for WL94D's ≈1.0. In that L1 norm the corresponding SVD16 is 0.891.
3. **Matched-norm gap at the field level, on the actual held-out incident distribution.** [MEAS]
   - Rank-16 non-negative material-level operators reach val field rel-L1 0.36 (energy 0.71, peak 0.35) to 0.54 (energy 1.09, peak 0.44).
   - The learned operators: 0.846–0.892 / 0.83–1.09 / 0.22–0.33.
   - At comparable energy (≈1), NMF16 beats learned T by about 0.31 in field L1 and about 0.11 in field peak.
   - So the WL94D learned S1 senders are far from what a rank-16 non-negative operator can represent on these fields.
4. **Rank 16 does impose a real ceiling for S1, unlike B.** [MEAS]
   - No tested rank-16 operator (SVD, NMF, L1-NMF, field-adapted L2 or L1, clipped, per-patch) gets val field L1 below about 0.36. Field peak stays ≤ 0.49 at energy ≈ 1, and ≤ 0.35 for the L1-optimal fits.
   - Field L1 ≤ 0.15 with peak ≥ 0.70 needs rank about 32 (svd32 0.353 / 0.706; L1 ladder 0.146 / 0.758). Rank 48 gives 0.049 / 0.847.
   - Floor B is essentially exact at rank 8–16 (field 0.012–0.017). S2 is intermediate: 0.16–0.30, and 0.14 at rank 32.
   - [HYP] A peak-targeted rank-16 objective was not tested. Whether one could exceed peak 0.5 without large L1 cost is open. The per-row mirror peaks form a near-permutation structure that is high-rank.
5. **Field adaptation buys nothing for S1 and does not explain the deficit.**
   - [MEAS] Panel incidence under point lights is near-single-bin per (light, patch): median 1 lit bin, top-1 share 0.94.
   - Over the population the fields span all 64 incident bins (99 % energy in 48 dimensions; 90 % of irradiance in 35 bins).
   - The field-adapted non-negative L1 bound (0.362) equals the operator-level L1 bound (0.364). The L2 field-adapted RRR16 is worse in L1 (0.80).
   - The opposite holds for S2: RRR16 gives 0.175 against svd16 0.300, because its span is narrower.
   - [INF] A material-level rank-16 operator cannot represent the S1 panel response on the actual incident distribution better than about 0.36 field L1. The learned operators' 0.85–0.89 is mostly not a field-distribution effect.
6. **Position conditioning does not lower the held-out bound.** [MEAS] A per-patch position-conditioned rank-16 fit (L1, non-negative) reaches 0.263 on train but **0.451 on val**, worse than material-level (0.362): it over-fits the training lights. Per-patch L2 gives 0.583 (val).

### What remains unresolved

- **Geometry-conditioned per-patch factors.** [INF]
  - With near-single-bin incidence, a per-(geometry, patch) rank-≤4 non-negative operator could reproduce any one item's four training lights exactly. Field-level bounds on training items are therefore 0 for an operator that knew the lights.
  - The generator is light-independent, so the relevant bound is over the light distribution. The material-level and per-patch-position values above are the matched proxies for that.
  - A geometry-conditioned bound, i.e. conditioning on the local light-placement distribution induced by geometry, was not computed.
- **Optimisation versus generator capacity.** Untested.
  - WL94D ran 1,000 steps; T lowered the train field L1 only 0.80 → 0.71–0.73 and the teacher loss 6.4 → 4.8.
  - Whether the generator (27-d patch features, 69→16 material MLP, shared bin embedding, softplus heads) can produce an NMF16-like a, c is unknown.
  - [HYP] The WL94D teacher normalisation (per-(l,n,c) bin mean) and the L2 objective favour level over lobe. That is consistent with T's energy gain (0.87 → 0.99) without peak gain (≈ 0.3), but it is not isolated.
- **The learned WL94D operators' Frobenius, mass and operator-peak** were not computed (needs model loads). The matched Frobenius comparison exists only for the WL92 frozen generators (0.88–0.89) and WL92A OWN (0.80–0.81, E4 T1 geometry), against NMF16 0.49–0.54.
- **Mapping to rendering is unmeasured.**
  - The bounds are relative to M64, not physical q. Exact M64 on the 16-point encoding is 0.36 from q on panel patches (WL92B, unweighted radiance rel L1; different norm) [MEAS].
  - The exact S1 sender plus exact receiver still leaves a CK2 flight floor of 0.53–0.94 rel L1 against e2_p [MEAS, 94D].
  - The E_H consequence of substituting a rank-16 bound operator at S1 (instead of exact M64, as in C2_srcS1) was not computed. So "rank-16 suffices for region C / CK2" is neither established nor refuted [HYP].
- **Receiver use** of the tied c (applied to E1 fields) was not bounded.
- **The W94 azimuthal mis-weighting** (`an94.py:19`) affects every WL94-series exitance or field number. Its impact on the bound metrics is ≤ 0.02 (field L1); its impact on the learned and render-derived values is unmeasured.
- **Scope:** one S1 material (M64 from a 16,384-sample MIS estimate; `nmf.npz` Mq assumed identical to `mat94.mq_ext()` rows 0–5, since both come from `diag92.local_matrices` with the same seeds [INF]); 96/24 items; 4 point lights per item; a flat frame for S2.

### Reference files
- `c:\Projects\ReST-GI\experiments\ecro_s94d\s94.py`
- `c:\Projects\ReST-GI\experiments\ecro_attr92\diag92.py`
- `c:\Projects\ReST-GI\experiments\ecro_mat94\an94.py`
- `c:\Projects\ReST-GI\experiments\ecro_cf92b\cf92b.py`
- `c:\Projects\ReST-GI\docs\worklogs\94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md`
- `c:\Projects\ReST-GI\docs\worklogs\92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md`
- Server results: `~/wl95_calc/wl95_calc.json`, `wl95_calc2.json`, `wl95_calc3.json`, `S1_ops.npz`
