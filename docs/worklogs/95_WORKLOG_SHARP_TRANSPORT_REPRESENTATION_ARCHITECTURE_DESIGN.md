# Worklog — WL95: Sharp Transport Representation Architecture Design (2026-11-15)

## Origin, scope and process

- **Brief.** "WL95 — Sharp Transport Representation Architecture Design" (user, 2026-11-15). It is a regular numbered batch: an independent architecture-design question, not a WL94-series evidence gap. The batch is design-only; the owner decides its class and name.
- **What was done.**
  - No repository code, checkpoint or production path was changed.
  - Nothing was trained, and no GPU job was run.
  - No successor was started.
  - The only computation was a CPU-only matched-norm calculation on LabServer63 (`CUDA_VISIBLE_DEVICES=""`), which read existing artifacts and wrote only to `~/wl95_calc/` (§9.3).
- **Process.** The user had opted into multi-agent orchestration (ultracode). The work ran as two workflows. Every agent was read-only.
  - WF1 (`wf_559cb30d-342`): 8 agents.
    - Six independent readers: the code contract; the canonical documents; WL86–88; WL89–92B; WL93–93D; WL94–94D.
    - Stage A error-contract auditor.
    - Stage B representability auditor, including the CPU calculation.
  - WF2 (`wf_af90c9f3-0cd`): 15 agents.
    - Four independent designers, each starting from a different first principle: local angular response; directional propagation; learned/physical ownership; dynamic-reuse contract.
    - Two adversarial critics per design: physics/representability/evidence, and cost/scaling/falsifiability.
    - A judge for Stages D and E.
    - A completeness and correctness critic checking against the brief.
    - A report writer that applied every valid critic correction (§9.4).
  - The supervising session gave the agents no preferred design. All agent outputs are kept verbatim in `docs/plans/wl95_sharp_transport_design/`.
- **Verification by the supervising session.**
  - Re-read `docs/Architecture.md:13, 21`, `docs/RESEARCH_ROADMAP.md:128`, `experiments/fnto_compiled83/closure83_dr.py:1-12` (eval = f·cosθ_o, anisotropy-capable lobes) and `experiments/ecro86/ecro86.py:57-67` (rigid-OBB box-only dirty test).
  - Confirmed the `an94.W` layout defect: `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, whereas the protocol is 4 rings × 16 sectors (`experiments/ecro86/protocol/ecro86.json:47`; `experiments/ecro86/pt86.py:153-156`).
  - Confirmed the `delivery_box` slice: `experiments/ecro_s94d/s94.py:309` uses `1130:1238`.
- **Labels.** Every statement carries one label:
  - [FACT]: verified in code or a document;
  - [MEAS]: a measured value, with its source;
  - [INF]: reasoned inference;
  - [HYP]: an untested hypothesis.

  No novelty is claimed.

**Verdict: ARCHITECTURE_SELECTION_PREMATURE (§8).** No primary hypothesis is recommended. Nothing is adopted or authorised.

---

## 1. Verified Current Architecture

### 1.1 Current ECRO / OWN pipeline

The deployed operator is WL92A OWN `own_s8601/8602`, which loads into the unchanged `model86.ECRO` (`experiments/ecro_stage93/train93.py:45-51`; `experiments/ecro_mat94/mat94.py:209-212`) [FACT].

**Bins** [FACT]
- bin = iz·16 + iφ, with iz = clamp(⌊4z⌋) and iφ = clamp(⌊(atan2+π)·16/2π⌋). That gives 4 rings × 16 sectors, each ΔΩ = 2π/64 ≈ 0.098 sr (`experiments/ecro86/pt86.py:153-156`; `experiments/ecro86/protocol/ecro86.json:47`).
- Patch order:
  - room 0–1087;
  - panel S1 1088–1129;
  - box A 1130–1183;
  - box B 1184–1237;
  - sphere S2 1238–1269 (`experiments/ecro86/scene86.py:29-39, 47-56, 64-71`).

**Source** [FACT]
- s[n,m,c] = (1/16) Σ_j Σ_l I·cos_j/d_j²·V_j·1[bin(w_j) = m]. This is the patch-mean direct irradiance per incident bin (`pt86.py:378-402`).
- The 16 points are one fixed draw shared by all patches (`pt86.py:114-117, 127`).

**Local sender** [FACT]
- u = c·s and Bq = a·u, so M_n = a cᵀ.
- M_n is rank ≤ 16, non-negative, per patch and per channel.
- c is tied between the sender and the receiver (`experiments/ecro86/model86.py:111-115, 120`).
- Bq is constant over each patch and each outgoing bin.

**Flight** [FACT]
- E1[n,m] = Σ_{κ=0..7} 1[hit ≥ 0]·Bq[hit, dbin]·rw_k, with rw_k = z_k·ΔΩ/8 and z_k = the ray's own cosine (`model86.py:40-46, 116-119`).
- K = 512 rays per patch, κ = 8 per incident bin. There is one fixed origin and direction pattern (`default_rng(86001)`), identical for every patch (`pt86.py:114-116, 353-376`).
- Invalid rays are not renormalised.

**Global term and readout** [FACT]
- y_g = P·Σ_n area_n·Q_n·u_n, rank 24, driven by u and not by E1.
- B = a·(u1 + y_g) (`model86.py:121-124`).
- OWN supervises the single term to t2 = e2_p + e2_o and the global term to t − t2 (`experiments/ecro_own92a/own92a.py:172-176`).
- Readout: hl = thr(x1)·B[nearest patch(x2), bin(x2→x1)], with no interpolation (`pt86.py:228-233`).

**Edit path** [FACT]
- `set_geom` → selective re-trace of dirty rays, bit-equal to a full rebuild because the offsets are fixed (`experiments/ecro86/ecro86.py:57-78`) → full source recompute → full `factors(gs)`.
- `factors(gs)` is a dense N² Ft followed by 3 global message rounds (`ecro86.py:82-87`; `model86.py:83-107`).
- Nothing in the learned factors is updated selectively.
- The dirty test is a rigid-OBB test on boxes A and B plus the moved-box origin (`ecro86.py:57-67`).

**Sizes and costs**
- [FACT] 735,664 parameters.
- [MEAS] Factor tables ≈ 43 MB; state 27 B/ray, 17.6 MB (`docs/worklogs/86_…md:128`).
- [MEAS] Per edit: 76–78 ms (`86_…md:214-215`). Selective state 21 ms vs full 26–28 ms (`86_…md:131, 214`).

### 1.2 Directional information-loss points (auditA §2)

All rows are [FACT] for the mechanism. The class column is [INF] unless it cites a measurement.

| # | Point | Mechanism | Class |
|---|---|---|---|
| L1 | Source s, angular | Point-light direction collapsed to a 0.098 sr bin | discretisation / aliasing |
| L2 | Source s, spatial | 16 fixed points, patch mean | fixed-pattern bias |
| L3 | a cᵀ | rank ≤ 16, non-negative, tied c | rank, or optimisation (not separated) |
| L4 | Bq table | constant over the outgoing bin and the patch | angular and spatial projection |
| L5 | Flight | 8 rays per bin, one shared pattern, a single realisation | sampling |
| L6 | rw cosine | the ray's own z times bin-constant radiance; worst in ring 0, where z ∈ [0, 0.25] | angular aliasing |
| L7 | Receiver | the same c acts on E1 | rank and discretisation |
| L8 | Global term | scene-wide rank-24 latent driven by u; no flight geometry | interface semantics. [MEAS] It compensates (WL86A Amendment 1; WL93D) |
| L9 | Generator inputs | form factors plus a global mean; ER state 69→16 | capacity / interface |
| L10 | Readout | nearest patch and bin, no interpolation | discretisation. [MEAS] An amplifier, not a defect (WL86A) |
| L11 | Sphere | patch-centre frame; below-horizon clamped to ring 0; M64 uses a flat frame (`quad87.py:15-16`) | interface semantics |
| L12 | M64 | fbar averaged uniformly over 8×8 incident cells, without cosine (`quad87.py:48-56, 240-242`) | discretisation; exact only if f is constant within the incident bin |

### 1.3 Frozen local rule ER

- [FACT] (`experiments/fnto_compiled83/closure83_dr.py:1-14`) ER is the WL83 learned-compiler output:
  - Lambert plus 7 GGX lobes, split into outer lobes and refracted inner lobes;
  - a t(w) mapping by η_e;
  - a piecewise-linear factor s(μ_i)s(μ_o);
  - 69 floats;
  - exact `eval`/`pdf`/VNDF `sample`;
  - `eval(wi, wo) = f·cosθ_o`.
- [FACT] The closure supports anisotropic lobes (au, av). Whether the S1, B and S2 states actually use au ≠ av was not checked.
- [FACT] ECRO re-learns a bin-projected, rank-16, tied copy of ER's once-scattered response inside its factor table.
- [MEAS] (WL87) ER reciprocity 1.2e-5; furnace 6e-7.

### 1.4 Evidence section A: physical and mathematical contracts

**ECRO transport equation** [FACT] (WL86:17-27)
- T = S_ER C_G and q = S_ER e, with paths truncated at D = 6 events.
- LOW = q + Tq is path-traced.
- b̂ = q + Tq + T·R_θ(G, z_ER)[e]. The last flight reuses the camera x1→x2 bounce.
- ER owns every evaluated scattering event. Geometry owns visibility. ECRO owns only events ≥ 3, and B excludes e2.

**Canonical K/C form** [FACT] (`docs/Architecture.md:25, 31-38`)
- i_a = e_a + Σ_b C_ab(G) b_b and b_a = K_a[i_a].
- K is linear, non-negative and passive at fixed geometry.
- Every segment has exactly one owner.
- Ballistic and scattered families are each read out once.

**Invariants**
- [FACT] ER is passive and reciprocal (§1.3).
- [FACT] ECRO is linear, non-negative and exact under superposition. It carries no passivity constraint (WL86:41).
- [FACT] No direct/indirect *response* branch is allowed: "Sharp … response demands sufficient directional information" (`docs/Architecture.md:21`).

**Unresolved mapping** [FACT]
- ECRO partitions transport by event count, not by K/C segment. No canonical document maps one onto the other (read_canon (B); WL86B:48).
- Every hypothesis in §3 inherits this gap.

**Reuse semantics** [FACT]
- Rule parameters are fixed across supported edits, with no retraining. Caches are invalidated together with their actual dependencies (`docs/Architecture.md:48-56`).
- Rule reuse (a fixed rule regenerates the state) differs from state reuse (WL86B:44). ECRO is rule reuse.
- Rigid / local deformation / bending / self-contact implications are mandatory for each major decision (`docs/RESEARCH_CENTRIC_TOPIC.md:20`).

**Reference split** [FACT] (`experiments/ecro86a/pt86a.py:73-200`)
- `field_src(split=True)` tallies by the event count k at which NEE fires: k = 2 → e2, k ≥ 3 → e3.
- It also splits by whether the k = 2 vertex has shape-level material id 1 (`mid == 1`, lines 175-180): `_p` = panel, `_o` = all others.
- The four tables sum to f. e2_o includes Lambert, floor B (glossy) and S2 senders.

### 1.5 Canon that binds any recommendation

- [FACT] (`docs/Architecture.md:13`):
  - ECRO is the resumed learned-global implementation base.
  - EXP/C3UR are "learning-free explicit transport comparators/diagnostics rather than the global method to be published".
  - "X1-style first-gather findings … do not justify converting the project to explicit GI".
- [FACT] (`docs/RESEARCH_ROADMAP.md:128`): the next decision must isolate "one falsifiable ECRO-lineage representation/training/ownership hypothesis". It must not "promote an explicit candidate as the project goal".
- [FACT] The owner question "is a learning-free global solve around ER a candidate or a baseline?" (`docs/plans/gm1_architecture_review/60_coordinator_synthesis.md:94-100`) is constrained by the 2026-11-13 decision, which leans towards "comparator". It is not formally closed.
- [FACT] Baseline-role terminology (AGENTS.md; `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:23-32`) defines four separate roles:
  1. Neural Full Recompute;
  2. Native Reuse;
  3. same-checkpoint F/S;
  4. path-traced re-render.

  Beating a path tracer is not superiority over Neural Full Recompute.

---

## 2. Failure and Evidence Attribution

### 2.1 Matched-norm representability (Stage B; S1 panel, material 1)

**Norms used in this subsection** [FACT] (definitions copied from `experiments/ecro_s94d/s94.py:sender` and `experiments/ecro_attr92/diag92.py:122-130`)
- **Op L1**: pooled flux-L1, Σ|A − M64|·zb_in·zb_out / ΣM64·zb_in·zb_out, with the true ring zb = ((d//16) + ½)/4.
- **Field metrics**: on the actual panel incident fields s of the 96 training and 24 validation items, field rel-L1 / energy / peak against Bm = M64·s, weighted with W94 (as WL94D coded it). True-W values are in auditB; they differ by ≤ 0.02 in L1.
- **Overlap**: (1 + mass − L1)/2, valid for non-negative A only.
- **Reference**: everything here is relative to the **M64 teacher**, not the physical q.

| Arm | Admissible for a cᵀ (non-negative, rank ≤ 16)? | Op L1 | Op peak | Op mass | Val field L1 / energy / peak | Overlap |
|---|---|---|---|---|---|---|
| zero | yes (trivially) | **1.000** [MEAS] | 0 | 0 | 1.000 / 0 / 0 | 0 |
| WL94D learned (own, D, T, R, P; both seeds) | yes | **0.961–1.044** [MEAS SN_*.json] | n/c | n/c | **0.846–0.892 / 0.83–1.09 / 0.22–0.33** [MEAS] | ≈ 0.4 [INF: WL92A OWN mass 0.80–0.84 on E4 T1, a different geometry] |
| svd16 (Frobenius-optimal) | no (signed) | 0.891 | .539 | .937 | .726 / .992 / .483 | n/a |
| svd16_clip0 | yes | 0.683 | .539 | 1.145 | .580 / 1.212 / .483 | 0.73 |
| nmf16_wl92 | yes | **0.603** | .518 | 1.038 | **.536 / 1.094 / .436** | 0.72 |
| nmf16_l1irls (L1-targeted) | yes | **0.380** | .462 | .689 | **.364 / .706 / .347** | 0.65 |
| fa_nmf16_l1 (field-adapted) | yes | 0.378 | .462 | n/c | .362 / .709 / .346 | n/c |
| per-patch position-conditioned non-negative L1 rank 16 | yes | mean 0.505 | n/c | n/c | train .263 → **val .451** / .592 / .317 | n/c |
| L1 non-negative ladder, random init: r8 / r16 / r32 | r8, r16 yes; r32 no (rank) | .660 / .464 / .233 | n/c | n/c | .618 / .416 / .208; .406 / .628 / .392; **.146 / .877 / .758** | n/c |
| svd32 / svd48 | no | .548 / .076 | .791 / .936 | n/c | .353 / .989 / .706; .049 / .990 / .847 | n/a |
| Contrast: floor B, nmf16 | yes | .032 | .965 | n/c | .016 / 1.001 / .978 | n/c |
| Contrast: S2, nmf16 / rrr16 | yes / no | .259 / .369 | n/c | n/c | .225 / 1.025 / .417; .175 / .996 / .742 | n/c |

All arm values without another citation are [MEAS, this calculation, `~/wl95_calc/wl95_calc*.json`]. n/c = not computed.

**What the table supports**
- [MEAS] The learned operators are 0.36–0.66 above admissible non-negative rank-16 operators in the same flux-L1 norm. At the field level, NMF16 beats learned T by about 0.31 in field L1 and about 0.11 in field peak, at comparable energy.
- [INF] So the rank-16 bound is **not** what currently binds the learned sender. Five candidate causes are **not** separated:
  - optimisation;
  - objective / normalisation (the WL94D teacher den);
  - generator parameterisation (27-d features, 69→16 material MLP);
  - sharing of `a` with the global decoder (B = a·(u1 + y_g), `model86.py:121-124`);
  - tying of c.
- [MEAS] The rank-16 ceiling would bind once that gap closed. No tested rank-16 operator reaches val field L1 below about 0.36, and peak stays ≤ 0.49 at energy ≈ 1. Field L1 ≤ 0.15 with peak ≥ 0.70 needs rank about 32.
  - This ceiling is **empirical over the factorisations tried**, not a proof.
  - A peak-targeted rank-16 objective was not tested [HYP].
- [MEAS] Floor B is near-exact at rank 16: field L1 0.012–0.017; 0.065 at rank 8 (svd8). S2 is intermediate: 0.16–0.30 at rank 16, 0.14 at rank 32.
- [MEAS] Panel incidence is near-single-bin per (light, patch): median 1 lit bin, median top-1 share 0.94. Over the whole population the fields span all 64 bins.
- [MEAS] Field adaptation buys nothing for S1: 0.362 vs operator-level 0.364.
- [MEAS] Position conditioning over-fits: per-patch 0.451 on val vs material-level 0.362. A geometry-conditioned bound was not computed.

### 2.2 Metric and code discrepancies (append-only corrections; prior worklogs are not rewritten)

1. **`an94.W` ring layout** [FACT]
   - `experiments/ecro_mat94/an94.py:19` sets `NB, NPHI, NZ = 64, 8, 8`, and `:32` computes W = ((d//8) + ½)/8·ΔΩ. The true layout is 4 rings × 16 sectors.
   - Effect: inside a ring, sectors 0–7 and 8–15 get cosine weights (iz + ¼)/4 and (iz + ¾)/4. In the grazing ring that is a 1:3 ratio. ΣW = π is preserved.
   - Where W is used:
     - the WL94D field and delivery metrics (`s94.py:287, 297-312`);
     - the CK2 flight floor (`s94.py:157`);
     - G_PHYS (`s94.py:129`);
     - WL94 exitance `an94.ex` (`an94.py:37, 104`);
     - WL94B (`experiments/ecro_exp94b/b94.py:47`).
   - It is neutral for Lambert outgoing radiance [INF], so the box-A and panel-face CK2 floor values are unaffected.
   - It biases non-Lambert numbers: the all-receiver 1.378, and S1 energy_ratio 1.71–1.75.
   - [MEAS] On the bound arms the bias is ≤ 0.02 field L1 and ≤ 0.04 energy. M64 S1 maximum albedo is 0.825 with the true weight and 0.899 with W94.
   - Its effect on the learned and render-derived WL94-series numbers is unmeasured.
   - The operator metric and T_ENERGY use the correct zb (`s94.py:179-181, 288`) [FACT].
2. **`delivery_box_rel_l1`** uses patches `1130:1238` (`s94.py:309`) [FACT], which is boxes A **and** B (108 patches), not box A alone as the WL94D wording implies.
3. **Unmatched WL94D comparison A** [FACT] (auditB §1c.1). WL94D's "operator ≈ 1.0 vs rank-16 Eckart–Young floor 0.46–0.52" (94D:109/147/178) compares pooled flux-L1 with per-colour flux-Frobenius.
   - [MEAS] In the matched L1 norm the zero operator is 1.000 and SVD16 is 0.891. The admissible matched references are 0.603 (NMF16) and 0.380 (L1-NMF16).
4. **Unmatched WL94D comparison B** [FACT] (auditB §1c.2). WL94D's "WL92 keeps 47–54 % of the mirror-bin flux; learned keep 21–33 %" compares an operator-level per-row peak with a field-level peak at the Bm argmax.
   - [MEAS] For the same NMF16 the two are 0.518 and 0.436. The 0.08 gap comes from the metric alone.
5. **WL94D "the 64-bin flight floor exists even for the exact sender"** is stronger than the evidence [INF] (auditA §3, bottom line).
   - The "exact sender" is M64(s). It carries L1, L2 and L12 as well as L4, and is read through the deployed fixed 512-ray state (L5, L6).
   - The decomposition is **OPEN** (§2.3).
6. **The CK2 panel-face correlation −0.334** [FACT] is a Pearson correlation over 9 patches for one light. The ideal-mirror footprint covers 3 of those patches [INF].
7. **Minor** [MEAS]:
   - T_SEM s8601 is 1.51e-7 in `audit.json` but printed as 7.6e-8 in 94D:42. Both are far below the 1e-5 threshold.
   - WL92's protocol wording "rows sum to the directional albedo" omits a z_bi/dΩ factor (`diag92.py:172`) [FACT].

### 2.3 Attribution by stage

**Local sender, learned**
- [MEAS] At the zero-operator level relative to M64 (§2.1). The learned panel Bq vs the exact physical q: rel L1 0.90–1.14, flux 0.16–1.07 (WL92:131; WL92B CF_E2_T1).
- [MEAS] S1 S_geo is 0.2–0.8 % (WL93D, `final93d.json`). The learned S1 operator is geometry-invariant to that level, and C1 swaps change nothing measurable.
- [INF] S_geo measures how much the operator varies across geometries, not what conditioning is worth.
- [MEAS] E_phys 0.75–0.82, and it is geometry-common (WL93D).

**Exact-sender substitution, M64(s) at the S1 panel**
- [MEAS] Untextured sharp renders improve:
  - WL93D C2_src, E2 T1 region C: 0.018–0.024 vs C0 0.040–0.077.
  - WL94 C2_srcS1, CK2 ROI_T: 0.031 / 0.019 vs 0.047 / 0.042, against noise 0.0084.
- [MEAS] It does **not** improve the textured CK2 renders (C2_src 0.031 → 0.063 with the checker; no arm is best on both seeds and variants) or the pooled single-flight table (+0.0052…+0.0079, WL93D).
- [INF] Co-adaptation is the recorded interpretation (WL93D COADAPTATION_LIMITED). Its mechanism is labelled an inference and was not isolated.

**Exact prefix + learned vs path-traced tail** [MEAS] (E2 T1 region C)
- ENCB (c = I, a = M64, R = 64, path-traced L₃₊; `diag92.py:204-210`): 0.0449 single draw, 0.027 pooled over 16 realisations.
- ORA2: 0.0127.
- ORA = X_M12 (same prefix + learned G): 0.030 / 0.033 (WL93C).
- [INF] The learned tail partly **cancels** the prefix deficit. So "exact sender + learned G is adequate" is a contaminated reading.
- [MEAS] After pooling, ENCB/EXP 0.027 still sits about 0.014 above ORA2. That residual is unattributed.

**Propagation**
- [MEAS] (WL90, GM1 W1/W2, exact q, pooled relations): the explicit-vs-oracle gap is the first-gather angular bin projection of L₁.
  - 97–102 % isolated; spatial −0.001 to −0.003; orders ≥ 2 only (P2plus) ≤ 0.0009; compounding ≤ 9 %.
- [MEAS] (WL91): pooled C3URX1 ≈ oracle, but the single-draw variance fails held-out (1/3) and utility rule (a) (3.1× slower at the 0.06 tier).
- [MEAS] (ECRO scene86, E2):
  - Sender versus receiver placement at equal rays and equal dense quadrature (C3UR − R0D, an explicit-relation factor, single draw) is +0.050 / +0.053 (WL88). This concerns the explicit relation, not the learned operator.
  - RAYSEND exact first-gather reads on the deployed R0 state: 0.059 vs EXP 0.049 (WL87).
  - RAY-vs-BIN patch RMS rose 1.3× on E2 T1 (0.50 vs 0.38) and 2.0× on E2 T0 (0.93 vs 0.47). Aggregate readings were mixed: RAY −20 / −20 %, BIN −16 / −26 % (WL86A Amendment 2).
- [INF] The GM1 projection attribution therefore does not carry over unchanged to the ECRO receiver-side regime, where sampling dominates. GM1 and ECRO E_H values are not cross-comparable (different scenes, masks and references).

**CK2 "exact-sender flight floor"**
- [MEAS] (`results/ecro_s94d/audit.json`): ratio 1.378 / 1.211 / 1.206; corr 0.713 / 0.716 / −0.334; rel L1 0.940 / 0.528 / 0.598; half-noise 0.127 / 0.058 / 0.069 (all non-panel / box A / panel face).
- [INF] Reference noise is 5–9× smaller than the error.
- Candidate causes, all **OPEN on CK2** (auditA §3.3), confounded in one light, one realisation and 9–54 patches:
  - C1: outgoing-bin L4;
  - C2: spatial L4/L2;
  - C3: incident L1 + L12;
  - C4: 16-point L2;
  - C5: κ = 8 single draw, L5;
  - C6: ring-0/1 cosine, L6;
  - C7: receiver; neutral for the Lambert masks, open for all receivers;
  - C9: W bias.
- [CPU geometry, INF] (auditA §3.2):
  - Panel incidence is grazing: cosθ 0.024–0.164, ring 0 only.
  - Panel irradiance varies 307×.
  - The ideal-mirror footprint covers 23 % of box A's panel-facing face. It falls in one column of 3 patches and is fed by 2 corner panel patches.
  - Receiver incident cosθ is 0.22–0.31, straddling the ring-0/1 boundary.
- [MEAS] G_PHYS's pooled ratio 0.953 on training items shows no generic over-delivery. It cannot exclude a generic misplacement floor, because label reliability is only 0.525.

**Sharp-mediated ≥ 3-event family** [MEAS] (WL94B `audit.json`; loss-normalised target units)
- On the CK2 panel face, e3_p is 0.755 of box e3 and e2_p is 0.580 of box e2.
- 61 % of B there is higher-order (WL94 post hoc).
- 69 % of the panel-face target is panel-mediated.
- [INF, mixed-source arithmetic, indicative only] e2_p accounts for about 0.39·0.58 / (0.39·0.58 + 0.61·0.755) ≈ 1/3 of panel-mediated panel-face transport. A design that fixes only the directly lit first flight addresses roughly that third.

**Global term and co-adaptation**
- [MEAS] The global term adds a geometry-blind +0.02…+0.06 (WL86A Amendment 1).
- [MEAS] OWN recovers per-term flux but not the panel lobe: OWN peak 0.21 vs NMF16 0.47–0.54 (WL92A).
- [MEAS] The LSPOT regression is attributed to the global term: X_L3 0.71 / 0.82 (WL92B).

**Material compression** [MEAS] (WL93D addendum; WL94)
- Learned local and global terms compress the CK2 albedo response: slope 0.28–0.63.
- Material diversification restores it but trades against the sharp baseline. The effect is seed-dependent (WL94A INCONCLUSIVE).

### 2.4 Evidence section B: supporting, weakening and inconclusive evidence per hypothesis (WL92–WL94D)

| Hypothesis | Supporting | Weakening | Inconclusive |
|---|---|---|---|
| **G1** (ER-continuous sender + explicit first gather) | [MEAS] Exact-sender render gains on untextured ROIs (WL93D C2_src; WL94 C2_srcS1). [MEAS] Pooled X1 = oracle in GM1 (WL91; cross-regime) | [MEAS] Single-draw variance and utility failure of its nearest relative (WL91). [MEAS] RAYSEND worse on the deployed R0 state (WL87). [FACT] Canon (§1.5). [INF] It leaves e3_p binned | Whether the CK2 floor is angular (C1) or sampling (C5): OPEN |
| **G2** (ELBR, ER-binned prefix + learned resolvent) | [MEAS] ENCB-type prefix pooled 0.027 on E2 T1 C. [MEAS] OWN per-term targets recover term flux (WL92A) | [MEAS] The R0-relation analogue of the prefix is 0.0449 single draw. [INF, cross-regime, GM1 only] WL90 P2plus ≤ 0.0009 suggests little ≥ 2-order headroom in GM1; this is the wrong quantity for CK2 e3_p | The M64(s) × Γ_s prefix is unmeasured. Λ_θ is unspecified |
| **E-L** (ECRO-lineage learned material-level sender, changed representation or objective) | [MEAS] The sender is directly supervisable (WL94D). [MEAS] Secondary gains: WL93A TQ peak 0.31→0.48; E2 render 0.076→0.046 | [MEAS] Zero-operator level (§2.1). [MEAS] WL94D NO_RELIABLE_IMPROVEMENT. [MEAS] T's gain is energy, not lobe (peak ≈ 0.3) | WL93C NO_SIGNAL; WL94A seed dependence; WL94C seed disagreement; delivery and E_H of the bound operators unmeasured |
| **H-0** (EXP/C3UR/C3URX1 null) | [MEAS] Explicit ≥ ECRO everywhere (WL87: 0 wins / 3 losses for ECRO). [MEAS] Pooled ≈ oracle (WL88 R0_INF; WL91) | [MEAS] Heavy-tailed single draws (WL87; WL91) | [FACT] Candidate vs comparator is an owner question |

### 2.5 Evidence section C: representation feasibility

- **Angular resolution**
  - [MEAS] The S1 panel half-energy radius r50 is 0.146 rad (WL86B).
  - [INF] π·r50² ≈ 0.067 sr ≈ 0.68 ΔΩ. The lobe's half-energy cone is therefore comparable to a single bin, so the bin projection is coarse rather than absurd.
  - [MEAS] The flux-X rank needed for 90 / 99 % energy is 25–33 / 46–59 (WL92).
  - [INF] More bins alone does not remove L1/L2/L12 or the sampling term L5, so it cannot be assumed to solve the problem.
- **Sharp lobe and energy**
  - [MEAS] S1 maximum directional albedo is 0.825 with the true weight.
  - [MEAS] L1-optimal rank-16 fits under-emit (energy about 0.7) and Frobenius fits are about 1.09. Peak and L1 must therefore be read together with energy (§2.1).
- **Propagation error accounting**
  - [FACT] E_H is an absolute-value metric. Step differences between arms are not additive error shares (WL91 caveat).
  - [FACT] Field rel-L1, operator norms and E_H are distinct norms (auditB §1c). Every attribution in §6 is therefore reported per norm.

---

## 3. Independent Architecture Hypotheses

### 3.0 Deduplication of the designer proposals

- [FACT] Four designers independently proposed an ER-continuous sender plus an explicit first gather:
  - LRS: a deterministic dense sender grid;
  - LCFG: receiver-point MIS with a stationary-point lobe proposal;
  - ELCF: light-side lobe-sampled photons;
  - DSR: sender-anchored shared origins.
- [INF] These are one architecture class with four estimators. They are factors of a smoke, not distinct architectures. The class is the WL86B H-0 / C3URX1 lineage (`experiments/gm1_c3urx1_91/cand91.py:36-110`: sender-side, exact per-ray L₁) with a different first-gather quadrature. It is called **G1** below.
- [INF] The only ECRO-lineage, source-independent candidate is **G2** (ELBR). It is materially distinct from G1 because it keeps a reusable binned first flight and a learned global object.
- Two reference rows are carried for comparison. Neither is a designer candidate:
  - **E-L**, which no designer specified;
  - **H-0**, the explicit null.

### 3.1 G1: ER-continuous sharp sender + explicit first gather (learning-free sharp first order)

**State**
- [HYP] Geometry with its BVH.
- Per-patch ER states (69 floats).
- A material-only peaked-sender class. It must be the prospective WL88 rule (median Π > 8: S1 37.95, S2 16.13, B 4.93 → S1 and S2 peaked) and must meet the WL86B admissibility criteria (prospective class rule, tallies summing to the reference, per-term tallied targets) [FACT for the rule; WL88, WL86B:222-227]. Otherwise it is an unjustified decomposition.
- A per-light first-flight cache of exact L₂ contributions from directly lit peaked senders, at receiver points or cells.
- A geometry-only relation for the remainder (C3UR-type).

**Operator I/O**
- [HYP] Inputs: light set, geometry, ER.
- Outputs:
  - L₂ from peaked senders at receivers, computed per light;
  - L₂ from non-peaked senders and orders ≥ 3, as patch × bin tables by explicit sweeps (H-0), or by an optional learned resolvent (UNSPECIFIED).

**Ownership**
- Local scattering: ER, evaluated exactly at the (incident, outgoing) direction pair of each estimator sample [FACT capability]. Convention: `eval` already contains cosθ_o (`closure83_dr.py:7`), so estimators must not apply that cosine again [FACT].
- Geometry-dependent propagation: explicit ray-traced visibility [FACT for capability].

**Sharp structure**
- [HYP] Exact directions for directly lit peaked senders. This removes L1, L3, L4 and L12 at that sender for the first flight only.
- [INF] Incidence on peaked senders that is itself indirect (e3_p: 0.755 of panel-face e3 on CK2) remains binned.

**Diffuse / nonlocal**
- [HYP] Binned explicit sweeps (as H-0), or a learned resolvent that is UNSPECIFIED.

**Reuse after edits**
- [INF] There is no learned transport state. Rule reuse is ER only.
- The relation is selective, as in WL88/WL91: selective = full, bitwise [MEAS GM1].
- The per-light first-flight cache is recomputed in full on any light change, and on any edit that moves a peaked sender or occludes its paths.

**Training targets and supervision**
- [FACT] None for sharp transport. ER is already learned. An optional learned resolvent would need separately tallied e3 targets [FACT: available through `pt86a.field_src`].

**Inference-time physical computation**
- [MEAS] For the nearest relative (C3URX1, GM1, WL91:256): 526,467 probe intersections plus 526,467 shadow rays per light, vs 20,320 for the q table.
- [INF] Estimator-specific dense variants scale ∝ L·(peaked area)·(sample density).

**Memory**
- [MEAS] C3URX1 (GM1): relation 20.8 MB + probe buffer 12.6 MB + 6.3 MB scratch.
- [INF, designer/critic arithmetic] Dense variants at 27 B/ray including segments: up to about 0.56 GB all-peaked at D = 1,024, growing as α⁻² beyond that.

**Cache invalidation** [FACT]
- The current dirty test is a rigid OBB of boxes A/B (`ecro86.py:57-67`). This applies to every row in §4, not only G1.
- Light move: full first-flight recompute [INF].
- Non-rigid edits: no invalidation rule exists [INF].

**Why it addresses the failure**
- [INF] It removes the measured learned-sender deficit by not learning the sharp first order. In GM1 it removes the first-gather projection mechanism (WL90).
- It does not address:
  - single-draw variance, the measured failure of C3URX1 (WL91);
  - the e3_p family;
  - canon admissibility (§1.5).
- **Not a demonstrated fix on CK2 or E2.**

### 3.2 G2: ELBR, ER-binned local response + reusable sender-side binned relation + learned ≥ 3-event resolvent

**State**
- [HYP] s, or q̂ = M64·s, at senders.
- A source-independent, sender-side, binned relation Γ_s (C3UR-type, about 650k rays) [FACT for C3UR].
- Learned Λ_θ: **UNSPECIFIED** (inputs, outputs, form and capacity).

**Operator I/O**
- [HYP] Sender table M64(s). Then L₂ = Γ_s gather of the patch × bin sender table with the exact M64 receiver.
- B = L₂ + Λ_θ(·) for events ≥ 3. Λ_θ's I/O is **UNSPECIFIED**.

**Ownership**
- Local scattering: full-rank M64 (ER bin projection) at both ends. This removes L3 and L7 but keeps L1, L2, L4, L6 and L12 [FACT about M64].
- Propagation: Γ_s (explicit ray-traced) for the first flight. Λ_θ (learned) for orders ≥ 3, supervised against an exclusive e3 tally [FACT `pt86a.py:175-199`].

**Sharp structure**
- [INF] Binned at 64 bins on both sides of the first flight. Its CK2 floor is unknown (OPEN; §2.3).
- [MEAS] The E2 T1 C prefix analogue on the R0 relation (ENCB) is 0.0449 single draw / 0.027 pooled.
- The M64(s) × Γ_s prefix is unmeasured. The C3UR 0.026 value was measured with EXP q, not M64(s), so it is unmatched.

**Diffuse / nonlocal**
- Λ_θ, UNSPECIFIED.

**Reuse after edits**
- [INF] M64 is material-only and reused. Γ_s is updated selectively (GM1: selective = full). Λ_θ factors are regenerated, assuming they are geometry-conditioned.
- Source-only update: [INF] the niche shrinks to a few ms once q̂ = M64·s.

**Training targets**
- [HYP] Per-term tallied e3 targets (event-exclusive). Reader-importance-weighted family scores, per the WL86B criteria.
- The form is unfrozen. WL93–94D-scale smokes on related learned terms ended NO_SIGNAL or INCONCLUSIVE [MEAS].

**Inference compute**
- [INF] M64 application is O(L·N·64·64·3). The Γ_s gather is O(N·K). Λ_θ is UNSPECIFIED.
- [MEAS, GM1, another scene] The relation update is 14.3 ms (WL89:350).

**Memory**
- [INF] M64 is about 6×64×64×3 floats. Γ_s is about 21 MB [MEAS GM1]. Λ_θ is unknown.

**Invalidation**
- As for G1: rigid OBB only; non-rigid undefined [FACT / INF].

**Why it addresses the failure**
- [INF] It removes the zero-operator-level learned sender (§2.1) while keeping ECRO lineage and source independence.
- [INF, cross-regime] Existing measurements predict little headroom for the learned part over explicit sweeps:
  - WL90 P2plus ≤ 0.0009 is a GM1 R_mirror bin-projection quantity, not CK2 e3_p importance;
  - sweeps take 11.6 ms (C3UR, GM1, WL89:353) vs the ECRO generator at about 12.5 ms (scene86, WL86). These are different scenes and sizes.
- **Not falsifiable until Λ_θ is specified (§7 Gate D).**

### 3.3 E-L reference row: ECRO-lineage learned material-level sender

- [HYP] Keep the ECRO form (WL86B H-E type), but give the sender one of the following:
  - a material-level (not geometry-conditioned) non-negative operator at rank ≥ 16 or 32;
  - an L1- or peak-aware objective;
  - possibly untied sender and receiver factors.
- State, I/O, ownership, reuse and training are as ECRO (§1.1), with the sender factor replaced. The precise form is **UNSPECIFIED** by any designer.
- [MEAS] Its decisive learning-free precondition, the delivery and E_H of the §2.1 bound operators substituted at S1, is unmeasured (auditB "not done").
- Falsified for the tested workload class if the bound operators themselves fail in delivery and E_H (§6 D1).

### 3.4 H-0 reference row: explicit null (EXP/C3UR/C3URX1)

- [FACT] Patch × bin explicit sweeps on the same or sender-side state. W_ray is computed at each ray's exact direction (65,536 MIS per material). The q table uses 16 points × 1,024 MIS.
- [MEAS] (WL87) E2 T1 C 0.049 single draw. Pooled 0.024–0.030 (WL87/WL88). C3UR 0.026.
- Per edit 89–95 ms. Source-only update 51–64 ms.
- [FACT] It is a comparator under the current canon.

### 3.5 Rejected alternatives

| Alternative | Reason (labelled) |
|---|---|
| **MEK** (learned equivariant lobe kernel) | [FACT] Duplicates ER, which is already learned and exact. [INF] Adds approximation error with no information gain. [FACT] Its built-in azimuthal isotropy conflicts with ER's anisotropic-capable lobes (`closure83_dr.py`); whether S1 is actually anisotropic is unverified. [FACT] Prior local-closure negatives: WL80C/80D/81. Not distinct from G1's local layer |
| **ALP** (LTC lobe-polygon integral + scalar pair visibility) | [INF] An estimator variant of G1. Scalar pair visibility is wrong exactly in visibility-change regions. Quadratic, source-dependent pair table (large-table risk). Its kill threshold was transplanted from an unmatched (M64-binned) norm. Kept only as a diagnostic idea |
| **VSA** (persistent virtual-source atoms for orders ≥ 2) | [INF] Unit error (stores Φ where intensity is required). A single atom cannot represent grazing incidence-dependent reflectance. Hidden per-frame readout visibility tracing. O(L·P^d) growth |
| **G1 as the method** | [INF] Inadmissible as the method under current canon by close analogy (Architecture.md:13 names EXP/C3UR as comparators and says X1-style findings do not justify explicit GI; roadmap:128 requires an ECRO-lineage hypothesis). Formal status is Gate A. [MEAS] Repeats C3URX1's measured failure mode. [MEAS] Forfeits the only measured learned advantage, the 3.5–3.9× cheaper source-only update (WL87). [INF] Addresses about 1/3 of panel-mediated CK2 transport |
| **G2 as currently specified** | [INF] The learned part is unspecified and its predicted headroom is small (cross-regime). [MEAS] Its R0-relation analogue measured 0.0449 single draw / 0.027 pooled (E2 T1 C); the M64(s) × Γ_s prefix is unmeasured |
| **Higher rank or more bins for the learned a cᵀ alone** | [MEAS] The learned operator is far above its rank-16 bound in the matched norm, so rank is not what binds now. [INF] More bins leaves L1/L2/L5/L12 in place. Retained only as the bound-operator diagnostic arm (§6) |
| **Learned beam / learned C propagation** | [MEAS] WL52: valid and stable, but 8–19× slower update and fails contact / visibility-transition locality |
| **Learned sampling density** | [INF] Cannot recover relations that are never hit. Whether it counts as "learned transport response" is an open owner question (`60_coordinator_synthesis.md:94-100`) |
| **Per-frame retrace (LPTG); DDT-type α⁻⁴ tables** | [FACT] Withdrawn by their own advocates (WL84A) |

---

## 4. Physical and Computational Trade-offs

### 4.1 Ten-criterion comparison

Cost and memory figures marked [INF] are designer or critic arithmetic, not measurements.

| Criterion | ECRO v0/OWN (status quo) | G1 | G2 (ELBR) | H-0 null |
|---|---|---|---|---|
| **1 Angular representability** | [MEAS] Learned S1 at the zero-operator level relative to M64. The rank-16 non-negative empirical ceiling is ≥ ~0.36 field L1 | [FACT] Exact ER at directly lit peaked senders. [INF] Binned for indirect incidence on peaked senders | [FACT] Full-rank M64 at both ends; keeps L1/L2/L4/L6/L12 | [FACT] As G2 (EXP q), or exact per ray (X1) |
| **2 Propagation fidelity** | [MEAS] 8 fixed receiver rays per bin; the deployed draw is in the bad tail (WL87) | [INF] Unbiased in expectation. [MEAS] Single-draw variance is the measured failure of the nearest relative (WL91). [INF, conflicting] Expected lobe hits per CK2 footprint receiver per realisation: about 1 (designers) vs 0.16–0.4 (critics); recompute from geometry (§7) | [INF] Sender-side binned. The CK2 binned floor is unknown | [MEAS] Pooled ≈ oracle; single draws heavy-tailed |
| **3 Material generalisation** | [MEAS] Albedo compression (CK2 slopes 0.28–0.63) | [FACT] Exact for any ER state. [INF] Sampling density is α-specific | [FACT] Exact final vertex. [INF] Λ_θ embedding risk | [FACT] Exact |
| **4 Dynamic geometry reuse** | [FACT] Weights reused; all factors regenerated | [INF] No learned transport state. Geometry- and source-dependent caches; full recompute on light moves and sharp movers | [INF] Rule reuse; Γ_s selective; Λ_θ regenerated | [MEAS GM1] Relation selective = full; sweeps global |
| — Rigid / deformation / bending / self-contact (all rows) | [FACT] The dirty test is rigid OBB, boxes A/B only (`ecro86.py:57-67`). [INF] Non-rigid edits have no invalidation rule. Only rigid box edits are covered | same | same | same |
| **5 Physical correctness** | [FACT] No passivity constraint (WL86:41); passivity not measured. [MEAS] Compensation (WL86A Amendment 1) | [INF] Unbiased first order (ER passive and reciprocal). Watch the cosine convention | [FACT] Exclusive e2/e3 tallies. [INF] Λ_θ only auditable; the ρ/(1 − ρ) bound is vacuous | [MEAS] Passive up to W_ray quadrature error (S1 directional-albedo error median 5.3 %, max 18 %; WL87) |
| **6 Training feasibility** | [MEAS] Seed-dependent; co-adaptation | [FACT] Nothing learned for sharp transport | [INF] One learned term with an unfrozen form. [MEAS] Related smokes ended NO_SIGNAL / INCONCLUSIVE | [FACT] None |
| **7 Inference / update cost** | [MEAS] Edit 76–78 ms (WL86), 69–78 ms (WL87); source-only 13–18 ms | [INF] Edit about 100–150 ms, anchored on EXP 89–95 ms (scene86) and C3URX1 124.7 ms edit-to-frame (GM1; cross-scene). [MEAS] Loses the source-only advantage. [INF] Scales ∝ L·(peaked area)·density | [INF] ≈ ECRO; the source-update niche shrinks to a few ms | [MEAS] EXP edit 89–95 ms; source 51–64 ms; PT 36 ms |
| **8 Memory** | [MEAS] About 43 MB factors + 17.6 MB state. [FACT] Dense Ft O(N²) as coded | [MEAS] About 21 + 12.6 MB (C3URX1). [INF] Dense variants up to about 0.56 GB, ×α⁻² beyond | [INF] ≈ ECRO; Λ_θ unknown | [MEAS] Relation about 17–21 MB |
| **9 Debuggability** | [INF] Entangled additive terms | [INF] High: every term physically tallied | [INF] High for the prefix; Λ_θ opaque | [INF] Highest |
| **10a vs path-traced re-render (PT role)** | [MEAS] Per frame 8.8–9.6× (descriptive; WL86). Matched-quality edit-to-first-frame 5.0–5.3× at the 0.03 level (WL87). [MEAS] The 9× belongs to the LOW + table structure, which H-0 shares (WL86B; WL87) | Not shown. [MEAS] Nearest relative: C3URX1 7.6–31× on the GM1 sharp tiers (WL91:288; C3URX1 only), but it failed utility rule (a) | Not shown | [MEAS] EXP 4.7–5.2× at 0.03 (WL87) |
| **10b Same-checkpoint F/S** | [MEAS] Selective 21 ms vs full 26–28 ms state rebuild (WL86) | [MEAS GM1] Relation S = F bitwise | [INF] As ECRO / H-0 | [MEAS GM1] S = F |
| **10c Neural Full Recompute (8DNA/RNA)** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |
| **10d Native Reuse** | not applicable | not applicable | not applicable | not applicable |

### 4.2 Risk of shifting cost into ray tracing or large tables

- [INF] G1 moves the sharp first flight into explicit ray tracing. All designers declared this, so it is not hidden.
- [INF] G1 becomes a per-source light or path tracer for that family when:
  - lights are many or moving;
  - lights are area or environment lights (no point-light NEE closure);
  - α → 0, where mirror continuation per atom is needed.
- [MEAS] Per light change it re-traces about 0.53 M probe plus 0.53 M shadow rays, at C3URX1 scale (WL91:256).
- Costs the critics found hidden or understated [INF]:
  - per-photon `fbar` (an MC histogram, `experiments/gm1_explicit87/quad87.py:85-110`), which is infeasible per photon;
  - per-ray f̄ tables for deferred orders ≥ 2 (about 0.5 GB);
  - 27 B/ray segment storage;
  - VSA's per-frame atom visibility at readout;
  - ALP's quadratic, source-dependent pair table.
- [FACT] G2's Γ_s is an explicit ray-traced relation of about 650k rays, the same class as ECRO's state. Its update cost belongs in G2's row: 14.3 ms in GM1 [MEAS, another scene].
- [FACT] The canon forbids "an exhaustive explicit per-surface full path tracer hidden under a neural name" (`docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86, 88`).

### 4.3 Initial construction cost vs per-edit update cost

**Offline**
- ECRO / E-L [MEAS / FACT]: targets for 1,024 geometries × 4 lights × 256 spp, plus training (WL86:129).
- G2 [INF from `quad87.py:223-242`]:
  - M64 per material: 64 incident bins × 64 sub-cells × 16,384 MIS ≈ 6.7e7 draws;
  - plus Λ_θ training (unspecified).
- H-0 [FACT]: W_ray at 65,536 MIS per material, plus the M64-class tables.
- G1: no training.

**Per scene build** [MEAS WL86]
- State 26–28 ms, plus relations.
- G1 adds per-light first-flight tracing.

**Per edit** [MEAS WL86; INF for G1 / G2]
- All rows are dominated by global recompute: the generator or the sweeps.
- Selective state saves only 5–7 ms.
- G1 adds a first-flight recompute whenever the edit touches peaked-sender paths.

### 4.4 Worst case and scaling assumptions

**Worst cases** [INF]
- ECRO: any edit regenerates all factors through the dense Ft O(N²) and 3 message rounds. Scaling with scene size is unmeasured (86:260).
- G1: cost ∝ L × peaked area × density; full recompute on every light change; per-atom continuation as α → 0.
- G2: ECRO-class generator plus the Γ_s relation update.
- H-0: J sweeps × N·K.
- All rows: the dirty fraction approaches 1 when a large mover crosses most rays.

**Scaling assumptions shared by every row** [FACT]
- compact point lights;
- planar peaked senders;
- rigid box edits;
- N ≈ 1.3k patches;
- α ≈ 0.05;
- nothing verified beyond scene86 / GM1;
- no sharp-mover, deformation or self-contact workload exists.

### 4.5 Evidence section D: system feasibility

- **Training targets** [FACT]
  - `pt86a.field_src(split=True)` provides exact event-exclusive tallies (e2_p / e2_o / e3_p / e3_o), so per-term targets are constructible.
  - [MEAS] Training-target noise is 0.10 rel L1 at 256 spp per lighting (WL86).
  - [MEAS] Target noise is heavy-tailed on some geometries (WL93A g00010).
  - [INF] Pooled L2 metrics need a heavy-tail guard.
- **Datasets** [MEAS]
  - Panel-mediated mass is about 3.4 % of the training target.
  - 4.2 % of items reach CK2-level panel exposure (WL94B).
  - [INF] Any learned sharp term faces an exposure problem that item resampling only partly fixes (1.23–2.84×).
- **Cache dependencies** [FACT]
  - Source encoding depends on lights and geometry.
  - State depends on geometry.
  - Factors depend on the whole geometry, through the global mean.
  - M64 depends on material only.
  - The G1 first-flight cache depends on lights, geometry and the peaked-sender class.
- **Admitted sharp held-out workload** [MEAS]
  - None exists. WL93D E6 was not admitted (SHARP_FAMILY 0.000 / 0.015 / 0.027 vs ≥ 0.08), and WL88 stopped unqualified.

---

## 5. Recommended Hypothesis

**None.** Recommending one now would mean selecting on theoretical expressiveness. Six reasons:

1. [INF] **G1 is inadmissible as the method under current canon**, by close analogy with Architecture.md:13 and roadmap:128 (§1.5). Its formal status is owner Gate A (§7). Until that ruling it is a comparator or oracle class.
2. [INF, cross-regime caveat] **G2's learned component is unspecified**, so its learned claim cannot be falsified.
   - Existing measurements in other regimes suggest little accuracy or cost headroom over H-0: WL90 P2plus (GM1), sweep vs generator cost (different scenes), and the WL87 source-update attribution.
   - Its R0-relation analogue measured 0.0449 single draw / 0.027 pooled on E2 T1 C. The M64(s) × Γ_s prefix is unmeasured.
3. [FACT] **E-L has no specified mechanism.** Its decisive learning-free precondition, the delivery and E_H of the bound operators, is unmeasured.
4. [INF] **The CK2 floor decomposition is open** (§2.3). It is the hinge between three readings:
   - a binned reusable first flight suffices (G2, E-L);
   - a continuous first gather is needed (G1);
   - the problem is sampling only (H-0 placement).
5. [INF] **The sharp-mediated ≥ 3-event family (e3_p) is owned by no well-specified mechanism.** G1 leaves it binned; G2 hands it to an unspecified Λ_θ.
6. [MEAS] **No admitted held-out sharp workload exists** (§4.5).

**Evidence-favoured constraint** (not a fix and not a selection) [INF]
- Current evidence does not support a geometry-conditioned learned rank-16 tied bin operator, as trained, as the owner of S1 local angular response. The basis is the matched-norm zero-operator level relative to the M64 teacher, and S_geo < 1 %.
- It does **not** establish that the rank-16 class is inadequate for delivery or E_H. That is untested.

---

## 6. Falsification Experiment

**Specified for owner decision. NOT recommended, NOT authorised and NOT started.**

**Declaration** [FACT for intent]
- This smoke is attribution-only and non-comparative. No external-baseline role (Neural Full Recompute / Native Reuse / F/S / PT) is measured, and no superiority claim follows.
- **Implementation boundary:**
  - a new diagnostics directory;
  - read-only use of `own_s8601/8602`, the scene86 / CK2 fields and the S1 operators;
  - no change to `experiments/ecro86/`, `model86.py`, checkpoints or production paths.
- **Batch class** [INF; the owner decides per AGENTS.md:3-11]: a dependent evidence-gap follow-up of WL94D's CK2 floor and WL95 Stage B. That makes it a suffix class (e.g. WL95A), not a regular number. No new learned mechanism is introduced.

### 6.1 Workloads (smallest valid)

- **W-a: CK2 "none", light 0** (grazing, ring 0). Reference: e2_p and e3_p from `CKF_none.npz` (2 × 4,096 spp halves; half-noise box A 0.058, panel face 0.069) [MEAS / FACT `mat94.py:288-299`].
- **W-b: one non-grazing CK2 light variant.**
  - It must lie inside the training rule `light_ok`, so the learned arms stay in-distribution.
  - It needs one new 2 × 4,096 spp `field_src(split=True)` reference.
  - It must be prequalified before freeze: SNR ≥ 4× on box A, under a geometry-only rule and a minimum-signal rule (WL88 / WL92A lessons).
  - If it fails, drop it and declare the CK2 result grazing-only.
- **W-c: E2 T1 / T0 region C** (WL86 REF-ER; WL92 ladder; ORA2 0.0127 / 0.0099).
  - [INF] A panel-tagged e2/e3 split appears to exist for E2 T1, because WL92 reports panel-family truth shares 0.18 / 0.36 / 0.46. Its exact file must be verified before freeze; otherwise this is a new render.
  - W-c targets the unattributed pooled residual (≈ 0.014 above ORA2).
- **GM1 W1 / W2: not rerun.** They are cited only as a known-floor reference in their own norm, never pooled with CK2 or E2.
- **Not run:** textured CK2. `field_src` takes the material from shape-level SMAT, which is unverified for textured variants (auditA C8). Declared out of scope.

### 6.2 Matched quantities (each regime labelled; never compared across norms)

- **Q_s (sender), reported in three separate norms, never mixed:**
  1. operator flux-L1 vs M64 (auditB norm);
  2. field rel-L1 / energy / peak / overlap vs Bm = M64·s;
  3. the same field metrics vs the physical q.
  - Weight: true W_T[d] = ((d//16) + ½)/4·ΔΩ. `an94.W` may appear only as a bridge to the old numbers.
  - **q-noise control:** q is an MC histogram (16 points × 1,024 MIS); its grazing-CK2 accuracy is unmeasured (0.013 is the GM1 value). Recompute q with redrawn MIS seeds and report its split-half noise.
- **Q_d2 (delivery, exactly 2 events, S1-mediated)**
  - Receivers: Lambert only.
    - Box A = mask 'T', patches 1130–1183 (54 patches), **not** 1130:1238 [FACT `mat94.py:31, 408`].
    - The panel face (9 patches) is reported separately.
    - The 3 footprint patches are descriptive only.
  - Statistics:
    - per-patch signed ratio with half-split CIs;
    - flux ratio;
    - signed and absolute rel-L1;
    - Pearson correlation only over the 54 box-A patches, with a bootstrap CI.
  - Reference: e2_p (material 1 only). S2-mediated delivery is out of scope and declared as such.
- **Q_d3 (panel-mediated ≥ 3 events)**
  - Explicit sweeps j = 2..4 with panel-last-vertex tagging, matching e3_p's event range up to the field truncation.
  - A single second panel flight may be reported only as a labelled lower-order bound, not as a matched quantity.
  - Scoping only; not a gate.
- **Q_r (integrated sharp interreflection)**
  - Rendered E_H on CK2 ROI_T and on E2 region C.
  - The table is composed as: arm's panel-mediated L₂ (its e2_p estimate) + path-traced e2_o + path-traced L₃₊.
  - Never use the seed-specific learned receiver or G (WL93D co-adaptation).
  - Report the signed error, the T0→T1 edit slope and the per-family noise_bound. Not PSNR.
  - **Q_r floor** = E_H of the path-traced oracle table (LOW + e2 + e3 tables through the same nearest-patch readout), computed on each workload. Arms are judged against this floor, not against zero.

### 6.3 Oracle hierarchy and arms

Every arm uses existing or clearly constructible capability. New code is declared as new.

| Level | Sender | Propagation | Isolates | Capability |
|---|---|---|---|---|
| L-learned | C0 own s8601 / s8602 Bl | deployed R0 (512 rays, seed 86001, `rw`); exact Lambert receiver | learned sender + existing propagation | existing (`s94.py:152-160` path) [FACT] |
| L-bound16a / 16b | `nmf16_wl92` / `nmf16_l1irls` | deployed and pooled | ECRO-lineage rank-16 capacity, measured in delivery and E_H | existing in `~/wl95_calc/S1_ops.npz` [FACT] |
| L-bound32 | rank-32 field-adapted non-negative L1 operator | deployed and pooled | rank 32 | **Not saved.** Must be regenerated deterministically by re-running the `wl95_calc2.py` ladder (seed 9500 + k·10 + c, CPU, about 1 min) with an added `np.savez` (new code, declared) [FACT]. `svd32` is signed and inadmissible |
| L-M64 | M64(s) | deployed and pooled | re-measures the WL94D floor; must reproduce 1.211 / 0.528 on box A | existing |
| L-q16 / L-q256 | exact q (`quad87.qtable`, P = 16 and P = 256 stratified) | pooled | M64→q16 isolates L1 + L12; q16→q256 isolates L2 | existing function; scene86 / CK2 plumbing new |
| L-q16-sub(a) | q at outgoing sub-bins a ∈ {4, 16} | pooled | L4 angular (C1) | new small code |
| L-X1 | exact per-ray L₁ (`cand91.radiance`: probe + shadow) | pooled R0, and pooled sender-side (C3UR-type) | full L4 / L6 removal at the read | port to scene86 / CK2 (new plumbing) |
| **L-DQ** (exact sender + higher-fidelity propagation) | exact L₁ from ER at a point light (eval convention per `closure83_dr.py:7`) | deterministic dense area quadrature over the panel plane × dense receiver points on each box-A patch, with light-shadow and panel→receiver visibility rays; no relation sampling | propagation reference below the physical reference | new code (ER eval + Mitsuba visibility). Defined **only** for exactly-2-event, point-light, planar-sender delivery. **No higher-fidelity propagation level exists for e3_p** |
| REF | path-traced e2_p / e3_p; REF-ER renders | — | physical reference | existing |

**Pooled propagation** means:
- ≥ 64 relation realisations, with origins and directions redrawn under a new seed per draw;
- source points also redrawn for the P = 16 arms;
- each arm also reports a per-draw distribution over ≥ 16 draws (median, 90th percentile, worst), with paired comparisons on common random numbers.

**L-DQ compute** [INF]
- The panel is about 1.68 m². At 1 mm cells that is about 1.7 M cells; × 54 patches × e.g. 64 points ≈ 5.8 G visibility pairs.
- Ray counts and wall time must be declared at freeze, with a convergence ladder (e.g. 2 mm → 1 mm → 0.5 mm).
- Per the standing GPU-utilisation requirement, every stage needs a full-scale GPU% and wall-time probe before freeze.

**Optional estimator factors** (G1 variants; only if the owner wants G1 characterised as a comparator; they never gate the decomposition):
- LRS: dense sender grid with grid-phase rotations;
- LCFG: stationary-point lobe proposal vs SAM-X1 uniform-area proposal (SAM-X1 is the mandatory control);
- ELCF: light-side photons with dense-cell receivers;
- C3UR / C3URX1 at matched total rays.

### 6.4 Error separation

1. **Local sender representation.** At pooled propagation, compare L-learned, L-bound*, L-M64 and L-q*. Read Q_s and Q_d2. Pooling removes the L5 confound.
2. **Directional propagation / flight.** With an exact sender, step through deployed → pooled → sub-bin → X1 → L-DQ, each against REF.
   - Steps are single-factor where possible; otherwise use a factorial of sender × {deployed, pooled}.
   - E_H and rel-L1 are absolute metrics, so step differences are **not** additive shares.
3. **Integrated sharp interreflection.** Q_r (against its floor) for each arm's L₂, plus Q_d3 scoping.

### 6.5 Controls

- **Non-S1-sender control (Lambert + B + S2).** Exact q on all non-S1 senders at pooled propagation, against e2_o on box A.
  - Expect agreement within noise plus the small B bin floor and the S2 flat-frame error [INF].
  - Pure Lambert-sender or floor-B-sender references would need a new material-tagged split (new code and a new render). The split today is `mid == 1` only [FACT `pt86a.py:175-180`].
- **Floor-B bin adequacy.** Restricted to the operator and field level (sub-bin a = 1 vs 16 on B). There is no path-tagged render reference for it.
- **Incident distribution.**
  - Grazing (W-a) vs non-grazing (W-b) vs the E2 lights; report ring shares.
  - Superposition sub-test: two point lights co-binned at the panel, binned vs continuous.
- **Directional discretisation.** The sub-bin ladder, plus a half-sector bin-frame rotation (L6 sensitivity).
- **Spatial.** P ∈ {16, 256}; redrawn vs fixed source points (L2 fixed bias vs noise).
- **Capacity.** Rank 16 (a, b) vs rank 32 bound operators.
- **Compute.** Every arm reports:
  - relation, probe, shadow and visibility ray counts;
  - ER evaluations;
  - warm uncontended ms (Dr.Jit compilation excluded and reported separately);
  - MB.
  - The total ray budget (pooled realisations × arms × workloads) is declared at freeze.
- **Reference noise.** An effect counts only if it exceeds 2 × half-split reference noise plus the arm's own split-half MC noise.

### 6.6 Validity gates (frozen before results; STOP on failure)

- **V0.** L-DQ vs e2_p on box A: flux ratio ∈ [0.95, 1.05], and rel-L1 ≤ 2 × half-noise + L-DQ's own MC noise; plus convergence-ladder stability.
  - Failure → STOP (a contract, reference or implementation defect). Do not attribute it to any loss point.
- **V1.** The non-S1-sender control passes at pooled propagation.
- **V2.** L-M64 on the deployed state reproduces WL94D box A: ratio 1.211, rel-L1 0.528. Lambert receivers are W-neutral.
- **V3.** W-b qualifies, or it is dropped.
- **V4.** q split-half noise on the CK2 panel is small relative to the M64→q difference. Otherwise the D2 reading is declared noise-limited.

### 6.7 Decision outcomes

All thresholds are proposals for owner freeze. Every outcome applies only **within the tested workload class** (S1, compact point lights, planar sender, rigid box edits).

- **D1, rank.** At pooled propagation, compare L-bound16 and L-bound32 against **L-M64** (same s, same frame).
  - If L-bound16 ≈ L-M64 within noise on Q_d2 and Q_r: rank is not binding for delivery or E_H on this workload. The learned deficit is then attributable to factors not separated here (optimisation, objective, generator parameterisation, shared `a`, tying). An H-E-type training hypothesis becomes the admissible ECRO-lineage route.
  - If L-bound16 is worse than L-M64 by > 2 × noise while L-bound32 is not: a learned material-level sender needs about rank 32 or a non-fixed-basis local representation. That would be a new regular-batch hypothesis, stated against ER ownership.
- **D2, incident and spatial projection.** L-M64 vs L-q16 vs L-q256 gives the C3 and C4 shares on CK2. Decision: whether the source encoding must carry a directional state at the sender input.
- **D3, sampling dominant.** With exact q, pooled propagation closes most of deployed-vs-REF, and sub-bin / X1 add nothing beyond noise.
  - Decision: a binned reusable first flight suffices given adequate relation sampling. G2 and E-L remain viable, and G1's continuous gather has no CK2 target.
  - The design question becomes relation placement and density (C3UR class), with a WL91-type per-draw rule.
- **D4, angular projection dominant.** Pooled q stays off, but sub-bin a = 16, X1 or L-DQ closes the gap.
  - Decision: a fixed 64-bin reusable flight has a CK2 floor. Candidates must carry sub-bin or continuous sharp-sender information through the first flight.
  - For the learned lineage this decides between source-independent sender-only refinement and G1-type per-source evaluation. That choice needs the owner rulings in §7.
- **D5, beyond the first gather.** Even L-DQ-consistent L₂ leaves Q_r above its floor by more than noise, and Q_d3 shows a large panel-mediated e3 residual.
  - Decision: the binding problem is the panel-mediated ≥ 3-event family. No first-flight design (all of G1) addresses it, and G2 must specify Λ_θ with flight geometry before it can be a hypothesis.
- **D6, defect STOP.** Applies only if pooled X1 ≈ L-DQ on Q_d2 but Q_r with that L₂ differs from the oracle-table floor by more than the noise of both.
- **Acceptance.** None. This smoke selects no architecture. It enables only D1–D5, together with owner Gates A and B.

---

## 7. Unresolved Questions

1. **OWNER GATE A.** Is a learning-free sharp first order (G1 class) admissible as part of the method, or only as a comparator?
   - [FACT] Canon currently implies comparator (Architecture.md:13; roadmap:128).
   - [INF] If G1 is admitted, the status becomes MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION (G1 vs H-0).
2. **OWNER GATE B.** Would a propagation partition by event order or sender class violate the spirit of Architecture.md:21? That partition means a dense or explicit first flight for directly lit peaked senders, and binned for C-mediated incidence on them.
   - [INF] The response rule would be shared; only propagation fidelity differs.
   - Gate B concerns the C-side partition. Gate A concerns admissibility as the method.
3. **EVIDENCE GATE C.** D1–D5 of §6.
4. **EVIDENCE GATE D.** Any learned global form (G2's Λ_θ, or E-L) must, before any training:
   - state a learning-unique claim, with predicted headroom over H-0 in a matched regime;
   - pass a per-geometry free-fit representability bound.
5. **Open items**
   - ownership of e3_p;
   - sharp movers, deformation, bending and self-contact (no workload exists; the dirty test is rigid box-only);
   - area and environment lights;
   - the α → 0 limit;
   - an admitted held-out sharp workload;
   - the impact of the W defect on learned and render-derived WL94-series numbers;
   - the event-count partition vs K/C placement of ECRO;
   - learned WL94D Frobenius / mass / operator peak (needs model loads);
   - receiver-side use of the tied c on E1 fields (not bounded);
   - external Neural Full Recompute: UNMEASURED (mandatory gate not run).
6. **Numerical disagreements to resolve by geometry-only CPU computation before any freeze**
   - **CK2 sender–receiver distance:** designer 0.25 m vs critic 1.1–1.5 m. [INF] The 0.15–0.29 m figure is the perpendicular panel–face gap (auditA §3.2). The footprint is fed from panel patches at y ∈ [−0.3, −0.1], and the receiver column is at the face's +y end, so path lengths far exceed the gap. The 0.25 m figure is probably the gap, not the path length.
   - **Expected lobe hits per footprint receiver patch per realisation:** about 1 vs 0.16–0.4 vs about 3–4 under different estimators.

---

## 8. Architecture Verdict

**ARCHITECTURE_SELECTION_PREMATURE.**

- [INF] No candidate is justified. No surviving hypothesis is both admissible under the current canon and supported by evidence that discriminates it from H-0 or E-L.
- [INF] The four designer "primary" recommendations collapse into one learning-free comparator class (G1). G2 and E-L are not yet falsifiable hypotheses.
- [INF] **Why not MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION.** The only fully specified class (G1) is inadmissible as the method under current canon, pending Gate A. G2 and E-L are not yet falsifiable. If Gate A admitted G1, the status would become MULTIPLE_CANDIDATES_REQUIRE_DISCRIMINATION (G1 vs H-0).
- [INF] The discriminating evidence is learning-free: the §6 smoke plus owner Gates A and B.
- [FACT] No design is adopted. No successor is authorised or started. No novelty is claimed.

### 8.1 Proposed living-document amendment (proposal text only; not applied)

- [FACT] A dated, append-only record is justified when WL95 closes, because a batch closing is a trigger under AGENTS.md.
- [FACT] Architecture.md holds contracts only, so its contract text stays unchanged.

PROPOSED text for `docs/RESEARCH_ROADMAP.md`, after the line-128 decision block:

> **WL95 (design-only, 2026-11-15): ARCHITECTURE_SELECTION_PREMATURE.** Four independent designs converged on one learning-free class: ER-continuous sender + explicit first gather (H-0 / C3URX1 lineage). Under the 2026-11-13 decision this class is a comparator, not the method. The ECRO-lineage alternative (ER-owned binned prefix + learned ≥ 3-event resolvent) lacks a specified learned form and a predicted advantage over explicit sweeps. Evidence-favoured constraint, not a fix: current evidence does not support a geometry-conditioned learned rank-16 tied bin operator, as trained, as owner of S1 local angular response (matched-norm zero-operator level relative to the M64 teacher; S_geo < 1 %). Whether the rank-16 class is adequate for delivery/E_H is untested. Open: the CK2 flight-floor decomposition, delivery/E_H of bound (rank 16/32) senders, ownership of the panel-mediated ≥ 3-event family, and owner Gates A (learning-free first gather: candidate vs comparator) and B (event-order propagation asymmetry vs Architecture.md:21). A learning-free suffix-class attribution smoke is specified in the WL95 worklog; it is not authorised.

Worklog content (not living-document content):
- the `an94.W` (8 × 8) defect;
- the `delivery_box` (boxes A + B) defect;
- the WL94D unmatched comparisons (§2.2).

---

## 9. Reproducibility / Sources

### 9.1 Evidence files read in full

Copied into the repository under `docs/plans/wl95_sharp_transport_design/` (design-study material, not adopted):
- `auditA.md`
- `auditB.md`
- `read_code.md`
- `read_canon.md`
- `read_wl86_88.md`
- `read_wl89_92.md`
- `read_wl93.md`
- `read_wl94.md`

### 9.2 Re-verified in this pass (read-only)

- `c:\Projects\ReST-GI\docs\Architecture.md:13, 21`: canon text quoted in §1.5 and §1.4.
- `c:\Projects\ReST-GI\docs\RESEARCH_ROADMAP.md:128`
- `c:\Projects\ReST-GI\experiments\ecro86a\pt86a.py:75, 170-199`: split by `mid == 1` at the k = 2 vertex; four tables e2_p / e2_o / e3_p / e3_o.
- `c:\Projects\ReST-GI\experiments\ecro_mat94\an94.py:19, 32`: `NB, NPHI, NZ = 64, 8, 8`.
- `c:\Projects\ReST-GI\experiments\ecro_s94d\s94.py:30, 309`: PANEL 1088–1129; `bx[1130:1238]`.
- `c:\Projects\ReST-GI\docs\worklogs\91_*.md:250-260`: C3URX1 ray and memory counts.
- Judge- and critic-verified:
  - `experiments/fnto_compiled83/closure83_dr.py:1-14`
  - `experiments/ecro86/ecro86.py:55-67`
  - `experiments/ecro_mat94/mat94.py:25-35, 288-299, 408`
  - `experiments/ecro_attr92/diag92.py:204-210`
  - `docs/worklogs/89_*.md:345-356`
  - `docs/worklogs/86_*.md:41`
  - `experiments/gm1_c3urx1_91/cand91.py:1-110`
  - `experiments/gm1_relation89/rel89.py:57-75`
  - `experiments/gm1_explicit87/quad87.py:85-110, 223-242, 268-338`

### 9.3 Stage B CPU calculation (LabServer63, `CUDA_VISIBLE_DEVICES=""`, writes only to `~/wl95_calc/`)

**Scripts**
- `~/wl95_calc/wl95_calc.py`: main; 15 min 52 s.
- `wl95_calc2.py`: field-adapted non-negative L1 rank 16 plus the rank ladder; 58 s.
- `wl95_calc3.py`: per-panel-patch position-conditioned rank 16; 9 min.
- Repository copies (md5 equal to the server files): `results/wl95_calc/wl95_calc.py` (f41ac9ef…), `wl95_calc2.py` (f5fa876f…), `wl95_calc3.py` (2c55528c…), with `wl95_calc*.json` and `S1_ops.npz`.

**Outputs**
- `~/wl95_calc/wl95_calc.json`, `wl95_calc2.json`, `wl95_calc3.json`.
- `S1_ops.npz` contains: `svd1..svd64` (signed), `svd16_clip0`, `nmf16_wl92`, `nmf16_l1irls`, `zero`, `rrr{4,8,16,24,32}_{W94,WT}` (signed L2), `rrr16_W94_clip0`.
- **Not saved:** `fa_nmf16_l1` and the L1 non-negative ladder, including rank 32.

**Inputs**
- `~/rest_x92/results/nmf.npz`: Mq [6,64,64,3]; Cn/An [3 mat, 3 ch, 64, 16].
- `~/rest_x94/results/select94.json`: 96 train / 24 val items.
- `s` from `~/rest_ecro86/data/<split>/<file>`. The panel is 1088–1129, with pmat == 1 asserted.
- Rows:
  - S1: 16,128 train / 4,032 val;
  - B: 98,304 / 24,576;
  - S2: 12,288 / 3,072.

**Sanity checks** [MEAS]
- svd16 Frobenius equals WL92 `local.json` exactly: S1 0.520 / 0.458 / 0.461.
- M64 S1 maximum albedo with the true weight is 0.8247, equal to WL94D T_ENERGY.

**Metric definitions**
- Operator L1 = Σ|A − M|·zb_in·zb_out / ΣM·zb_in·zb_out, pooled over (in, out, c), with zb = ((d//16) + ½)/4 (as `s94.py:314-323`).
- fro: per colour, on the flux X (as `diag92.py:122-130`).
- op peak: per-row true-peak retention.
- mass = ΣA·ZZ / ΣM·ZZ.
- Field metrics, with Bl = S·A and Bm = S·M over the pooled items:
  - rel-L1 = Σ|Bl − Bm|·W / ΣBm·W;
  - energy = ΣBl·W / ΣBm·W;
  - peak = ΣBl[argmax Bm] / ΣBm[argmax Bm].
  - Each is reported with W94 (= `an94.W`, as coded) and with WT = zb·ΔΩ (true layout).
- Overlap = (1 + mass − L1)/2, for non-negative A.

**Limitations** [FACT]
- One S1 material.
- M64 is a 16,384-sample MIS estimate. `nmf.npz` Mq is assumed equal to `mat94.mq_ext()` rows 0–5 [INF: same `diag92.local_matrices` seeds].
- 4 point lights per item.
- Flat frame for S2.
- All numbers are relative to the M64 teacher, not physical q.

### 9.4 Critic items not adopted

- **Rejected: none.** Every valid correction from the completeness critic (A1–A11, B1–B14, C1–C10, D, E) was applied.
- Adopted with a modification:
  - **A5.** The critic's M64 offline cost "64 × 64 bins × 64 cells × 16,384 MIS" over-counts if read literally. Each MIS draw lands in one outgoing bin, so the count is 64 incident bins × 64 sub-cells × 16,384 ≈ 6.7e7 draws per material [INF from `quad87.py:223-242`, per read_code (5)].
  - **C8.** The 1 mm L-DQ cell size is kept as one rung of a declared convergence ladder, not as a fixed resolution.
  - **C9.** For W-c, a panel-tagged E2 split is inferred to exist from the WL92 family shares. It is not asserted; it must be verified before freeze.
