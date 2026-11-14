# WL93 → WL93D (+ checkerboard addendum): ECRO local-operator interventions and attribution

Read-only. No files were edited and nothing was trained or run on a GPU. Paths are relative to `c:\Projects\ReST-GI\`. WL = `docs/worklogs/`.

## 0. Common frame across all six documents
- [FACT] Every arm uses table B = S_θ(s) + G_frozen(s). G is the WL92A own_s<seed> global term (3–5 events). It is frozen and identical across arms. Only the single-flight local factors a_r·(c_r·flight(a_s·(c_s·s))) are trained. Source: WL/93_…:42-50.
- [FACT] All smoke batches (93A/B/C) share one contract: 128-geometry subset, 2,000 steps, seeds 8601/8602, B0 targets. Global term frozen. Source: WL/93A_…:66-70, WL/93B_…:34-37, WL/93C_…:54.
- [FACT] Parameter counts: tied 405,424; untied = tiedcm 487,952. Source: WL/93_…:155.

## 1. Interventions and measured effects

### WL93: source-stage supervision × untying, full-scale design
- [FACT] Design: 6 arms (T0/TQ tied, U0/UQ untied, C0/CQ tiedcm) × 2 seeds. TQ/UQ/CQ add the source-stage loss w·‖(Bq−q)/den_q‖², with Bq = a_s·(c_s·s). Source: WL/93_…:60-74.
- [MEAS] It stopped at the identifiability preflight, so no arm was trained. Validation encoding-through-ER vs q = 0.508; the bound was 0.485/0.483 (0.5 × learned 0.970/0.966). E2 passed at 0.362. Label: STOPPED-INCONCLUSIVE. Source: WL/93_…:194-202.
- [MEAS] Post hoc: the pooled failure is concentrated in g00010 (19.7 % of panel q mass, irreducible 0.99, q-vs-q 0.91) and g00039. 52 of 55 geometries pass individually. Correlation of the irreducible term with target noise r = 0.92. Source: WL/93_…:209-216.
- [FACT] Pre-freeze defect: a copy-averaged tiedcm stayed identical to tied under Adam (bit-for-bit C0 ≡ T0). It was replaced by an additive zero-output head2. Source: WL/93_…:51-56.

### WL93A: target reliability + tied T0 vs TQ smoke
- [MEAS] Target noise is spatial (P-dominated). Train-representative N_b: B0 0.095 → B2 0.028 → B5 0.0077. Raising Mq ×4 gives only 1.1×. g00010 still shows 0.66 noise at B5; the other 54 geometries are ≤ 0.06. Source: WL/93A_…:137-145.
- [MEAS] H_TARGET NOT SUPPORTED: enc vs stored q 0.508 against enc vs B5 0.495 (Δ 0.013). WL93A explicitly corrects WL93's post hoc noise reading: g00010 is a genuine encoding/operator discrepancy (M(enc) 3.77 vs its own noise 0.66). Source: WL/93A_…:158-165.
- [MEAS] TARGET_RELIABLE: RELIABLE (budget B0, δ 0.0987). Source: WL/93A_…:166-169.
- [MEAS] Primary dM(TQ−T0) = +0.051 / +0.125, so the verdict is NO_SIGNAL_AT_SMOKE_SCALE. Whole +4.6–5.4 %, single +7.1–7.3 %, so NOREG fails. Source: WL/93A_…:185-201.
- [MEAS] Secondary metrics improve:
  - rel L1 0.899 → 0.738 / 0.927 → 0.790;
  - peak retention 0.31 → 0.48;
  - E2 sharp M 0.83 → 0.59;
  - render E2 T1 region-C E_H 0.076 → 0.046 / 0.042 → 0.035.
  
  No fitting of target noise: fit to own target ≈ M vs B5. Source: WL/93A_…:192-213.
- [MEAS] Post hoc: g00010 accounts for 78 / 88 % of the pooled change, and TQ is worse there. Without g00010, dM = −0.053 / −0.069. Source: WL/93A_…:204-206.

### WL93B: source/receiver factor separation (untied) smoke
- [MEAS] Validation WHOLE: T0 .0762/.0771, TQ .0803/.0806, UQ .0770/.0775, CQ .0802/.0806, U0 .0735/.0737. Source: WL/93B_…:131-139.
- [MEAS] SRC: TQ .738/.790, UQ .737/.808, U0 1.067/1.092. Source: WL/93B_…:131-139.
- [MEAS] UQ−TQ WHOLE = −0.0032 / −0.0031 (CI excludes 0), better on 55/55 geometries. That recovers 79 / 87 % of the TQ regression. UQ reaches parity with T0, not a gain. Source: WL/93B_…:147-166, 221-223.
- [MEAS] Label: SHARING_SIGNAL. Source: WL/93B_…:158-160.
- [MEAS] The capacity control was inert: CQ−TQ WHOLE ≈ −7e-5 and C0−T0 ≈ 1e-5. Capacity is ruled out only for this control form. Source: WL/93B_…:155-156, 224-227.
- [MEAS] U0 (untied, no supervision) is non-physical:
  - SRC worse than init;
  - receiver on physical flight 0.119 / 0.177;
  - E1 LSPOT 0.044 / 0.047 fails F4.
  
  Source: WL/93B_…:205, 228-232.
- [MEAS] Learned source and receiver panel operators differ by 0.57–0.68 under UQ. Source: WL/93B_…:198.
- [MEAS] LPAT 2×2 cone checker: all arms smear the reflected spot into a vertical strip. Region-C E_H is 0.48–0.61, and arm differences are inconsistent. Source: WL/93B_…:208-213.
- [FACT] LPAT was a 2×2 checker because the interface allows ≤ 2 lights. Source: WL/93B_…:80.

### WL93C: dual-stage physical supervision (tied) + M64 oracle
- [FACT] Oracle composition: ORA = M64(flight(M64(s16))) + G. It equals WL92B X_M12 to 1e-7. Source: WL/93C_…:28, 95-96.
- [MEAS] Validation results:
  - ORA: single 0.1033, whole 0.0776 / 0.0774;
  - ORA_PHYS floor 0.0998;
  - ORA_SRC whole 0.0796 / 0.0795 (worse);
  - ORA_RCV whole 0.0756 / 0.0772.
  
  Source: WL/93C_…:105-112.
- [MEAS] Gate FEASIBLE vs TQ. Limiting factor LOCAL_OPERATOR: ORA single vs T0 is +0.0010 / −0.0010. The pooled tie comes from g00010 (ORA single 0.464). Source: WL/93C_…:115-120.
- [MEAS] Sharp oracle renders, region-C E_H:
  - E2 T1: 0.030 / 0.033 vs T0 0.076 / 0.042;
  - E5 T0: 0.015 / 0.014.
  
  Edit slope 0.995 / 1.000 vs T0 0.57 / 0.75. Source: WL/93C_…:133-141.
- [MEAS] The exact composition removes the single-flight level deficit (−1.2 % → +0.15 %). No compensation conflict with G. Source: WL/93C_…:123-127.
- [MEAS] TR (receiver-only supervision): receiver −0.0023 / −0.0030, whole +0.0009 / +0.0013, and worse E2 renders. Source: WL/93C_…:149-173.
- [MEAS] TQR (source + receiver): keeps 85–91 % of TQ's source gain. The receiver returns to T0 level (+0.0003 / +0.0005 vs T0). Whole vs TQ −0.0004 / −0.0006, not significant. Source: WL/93C_…:150-162.
- [MEAS] Label: NO_SIGNAL_AT_SMOKE_SCALE (criterion r fails for both seeds). Source: WL/93C_…:164.

### WL93D: frozen local-operator ownership (no training)
- [FACT] Arms (single-flight local uses only; geometry, rays, encoding and G fixed):
  - C0: deployed operator;
  - C1_src / C1_rcv / C1_both: the partner geometry's learned operator;
  - C2_src / C2_rcv / C2_both: M64, with C2_both = X_M12 = ORA.
  
  Families: own (WL92A) and T0 (WL93A). Source: WL/93D_WORKLOG_ECRO_FROZEN…:63-71, 29-36.
- [MEAS] Validation single-flight Δ vs C0:
  - C1_both +0.0007 to +0.0021;
  - C2_src +0.0052 to +0.0079 (worse);
  - C2_rcv −0.0009 to −0.0035 (better);
  - C2_both −0.0020 to +0.0010.
  
  Source: WL/93D_…FROZEN…:174-181.
- [MEAS] Renders, region-C E_H:
  - C2_src best everywhere: E2 T1 0.018–0.024 vs C0 0.040–0.077; E5 T1 0.013–0.018 vs 0.025–0.033.
  - C2_rcv worse, e.g. E2 T1 0.085–0.093 / 0.056.
  - C1_both within noise.
  
  Source: WL/93D_…FROZEN…:185-192.
- [MEAS] E6 held-out was not admitted. All three candidates failed SHARP_FAMILY (0.000 / 0.015 / 0.027 vs ≥ 0.08), so conclusions are limited to the inspected E2/E5. Source: WL/93D_…FROZEN…:136-141.
- [FACT] Disclosed exposure: the dev run printed Task A subset values (S_geo ≈ 0.003) before the evidence run. Criteria were frozen earlier. Source: WL/93D_…FROZEN…:118-121.

## 2. What WL93D established about ownership and geometry specialisation

- [MEAS] S_geo < 1 % holds for S1 panel, S2 sphere and LAM, but not for B floor.
  - S1 workload S_geo: own 0.0027–0.0035, T0 0.0060–0.0081. Canonical: own 0.0020–0.0024, T0 0.0044–0.0056.
  - S2: 0.0007–0.0028.
  - LAM (control): 0.008–0.017.
  - **B floor: 0.032–0.048.**
  - G2 needs ≥ 0.25 and fails everywhere.
  
  Source: `results/ecro_local93d/final93d.json` (`/functional/families/*/*/*/per_seed/*/S_geo`); WL/93D_…FROZEN…:154-157.
- [MEAS] S1 numbers:
  - E_phys 0.75–0.82, and the geometry-mean operator error E_common is 0.788–0.807. The error is geometry-common.
  - D_geo 0.040–0.084 overall, 0.09–0.14 on edit pairs. That is 3–6× LAM but 3–6× below D_seed (0.26–0.32).
  - G1 fails (< 0.10).
  
  Source: WL/93D_…FROZEN…:154, 159, 195-196.
- [MEAS] G3 fails:
  - No region-C cell differs beyond its margin; e.g. own s8601 E2 C1_both 0.0836 vs C0 0.0769 with margin 0.0126.
  - The validation |C1_both−C0| ≥ 2 SE holds only for s8601 in each family.
  
  Source: `final93d.json` `/criteria/*/G3_*`; WL/93D_…FROZEN…:197-199.
- [MEAS] Task C, incident subspace:
  - Receiver use: 97 % of evaluation energy is in the training span, with in-span error 0.31–0.34.
  - Source use: 65 % of the energy is out of span, with error equal in and out of span (0.58–0.69 / 0.59–0.63).
  
  Source: WL/93D_…FROZEN…:164-169.
- [MEAS] Labels for both families: REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED. GEOMETRY_SPECIALIZATION_SUPPORTED is not met. COADAPTATION is triggered by "C2_both not better than C0 by 2 SE for both seeds"; the interaction term holds for s8602 only. Source: WL/93D_…FROZEN…:200-206; `final93d.json` `/criteria/{own,T0}/labels`.
- [FACT] Interpretation recorded in WL93D:
  - H_G not supported as a material contributor. This rejects the IDTS causal premise for this lineage but does not test IDTS as an architecture.
  - H_R supported.
  - H_C supported. The compensation mechanism is labelled an inference, not isolated.
  
  Source: WL/93D_…FROZEN…:225-231.
- [INF] Ownership reading:
  - The local operator's sharp failure is not "geometry information owned by the local operator". It is a geometry-invariant mis-shaped response, mostly within the training span on the receiver side.
  - The learned receiver and global terms are tuned to the learned source's errors, so a single-use physical substitution moves table and render metrics in opposite directions.

## 3. The CK / CK2 checkerboard scene (post hoc qualitative addendum; non-numbered, no labels)

### CK (revision 1, withdrawn)
- [FACT] Back-wall checker on E2 T1, 25 cm cells, variants ck38 / ck75, plus a post hoc 1 m variant ck38L (amendment A4). Source: `experiments/ecro_local93d/addendum_checker/protocol_checker93d.json`:5, 11.
- [MEAS] The S1 lobe at about 1.8 m did not resolve the cells even in the reference: LOW parity 0.84 / 0.85 vs ideal 0.43 / 1.14. A geometric-only visibility rule had passed. Source: WL/93D_WORKLOG_ADDENDUM…:14-17.
- [MEAS] Withdrawn values, recorded only in a git commit message (commit 8c11bea): at 1 m, ck75 learned-receiver slope 0.17–0.31.
- [FACT] The owner rejected CK and ordered a scene change, verified on actual renders. Source: WL/93D_WORKLOG_ADDENDUM…:19-22.

### CK2 (revision 2) definition
Source: `experiments/ecro_local93d/addendum_checker/scene_ck2.json`; WL/93D_WORKLOG_ADDENDUM…:35-41.
- [FACT] Geometry and light:
  - tag CK2, light set LCK2;
  - box A at c (−1.08, 0.60, 1.00), half sizes (0.20, 0.45, 0.45), yaw 0.15, inside the training box rule (`geom_ok`). Its panel-facing face is about 0.22 m from the panel plane (x = −1.5).
  - box B at (1.4, 1.5, 0.5), out of view;
  - one point light at (−1.45, −0.6, 1.7), rgb 30, inside the training light distribution (`light_ok`);
  - camera (0.3, −1.6, 1.3) → (−1.3, 0.45, 1.0), fov 32°, 512 px. The camera is not a model input.
- [FACT] Checker: all six faces of box A, 3×3 patches per face (54 patches), one cell per 30 cm patch, parity (iu+iv). Source: `protocol_checker93d.json`:91.
- [FACT] The texture is a per-patch material-id map PMAT, consumed identically by three paths. Source: `experiments/ecro86/model86.py`:95 (`self.mat_mlp(self.mat_states[gs.mat])`).
  - learned ECRO: `mat_states[PMAT]`;
  - M64: `Mq[PMAT]`;
  - reference renderer: run-time patched `pt86.Ctx.trace`.
- [FACT] Variants:
  - bx38: Lambert 0.3 / 0.8, new rows, outside the training Lambert range 0.5–0.7;
  - bx75: ids 3 / 4 (0.7 / 0.5), 1.4:1 contrast;
  - "none": untextured counterpart rendered with common random numbers.
  
  Source: `protocol_checker93d.json`:94-95.
- [FACT] Why it exists:
  - It was a post hoc owner request to test whether a reflected albedo pattern seen via the sharp S1 panel is reproduced by the frozen model and the C2 arms.
  - CK failed because the panel lobe did not resolve the cells.
  - CK2 places the box at the minimum distance the training rules allow, with the largest face, so the checker is visible in the reference reflection. This was iterated on probe renders p01–p05.
  
  Source: WL/93D_WORKLOG_ADDENDUM…:19-24, 43-52; `protocol_checker93d.json`:86.
- [MEAS] Preflight passed 9/9. ROI_T = 58,035 pixels / 15 patches. Source: WL/93D_WORKLOG_ADDENDUM…:61-75.

### CK2 results
- [MEAS] Reference: ideal-mirror correspondence corr 0.81, slope 0.46–0.47. Higher-order ratio h: 0.83 / 1.36 (bx38) and 1.36 / 1.14 (bx75). Source: WL/93D_WORKLOG_ADDENDUM…:90-95.
- [MEAS] Arms place the texture response on the correct cells (corr 0.70–0.97) but compress its amplitude. Source: WL/93D_WORKLOG_ADDENDUM…:109-114, 133-137.
  - Response slope 0.28–0.63 (bx38) and 0.41–0.62 (bx75).
  - C0 s8601 hl on the 0.3 cells is 1.00, against the reference 0.83.
- [MEAS] The checker raises ROI higher-order E_H for every arm:
  - C0 0.047 → 0.076;
  - C2_src 0.031 → 0.063;
  - with the checker, no arm is best on both seeds and both variants.
  
  Source: WL/93D_WORKLOG_ADDENDUM…:138-141.
- [MEAS] Table-level ratios. Source: WL/93D_WORKLOG_ADDENDUM…:120-125, 142.
  - The M64 receiver reproduces the albedo ratio exactly in S (0.600 / 1.599 / 1.399).
  - The learned local operator compresses it: S 0.78–0.94 vs 0.6, 1.28–1.35 vs 1.6.
  - The learned global term compresses it too: 0.77–0.88 / 1.25–1.28. G carries 49–52 % of B on these patches.
  - So even C2_rcv's total B ratio is only 0.68–0.75 / 1.43.
- [HYP] The addendum leaves the cause of the albedo compression untested. Candidates it names: albedo as 1 of 69 state entries through `mat_mlp`, and per-object fixed materials in training. Source: WL/93D_WORKLOG_ADDENDUM…:155.

## 4. Attribution: target noise vs representation vs co-adaptation

**Target noise: largely not the cause; it acts mainly as a metric dominance problem.**
- [MEAS] WL93's post hoc noise attribution (r = 0.92) was refuted by WL93A against B5 (H_TARGET NOT SUPPORTED). g00010's discrepancy is genuine. No fitting of training-target noise was detected. Source: WL/93A_…:162-165, 200.
- [MEAS] However, g00010's heavy-tailed target (0.66 noise at B5) dominates the pooled L2 metric (78–88 % of the change). That is a disclosed protocol limitation. Source: WL/93A_…:205-207.

**Representation / optimisation: the primary attribution.**
- [MEAS] WL93C LOCAL_OPERATOR: the learned tied operator cannot reach M64 under the objectives tried. Under joint source+receiver supervision the receiver gain vanishes; WL93C leaves the cause (optimisation conflict, objective scaling or rank-16 tied capacity) open. Source: WL/93C_…:116, 194-202.
- [MEAS] WL93D H_R: S1 E_phys ≈ 0.80 and geometry-common; receiver error mostly in span. Source: WL/93D_…FROZEN…:229.
- [MEAS] The deployed 16-sample source encoding is a separate representation floor: enc M 0.25 train-rep, rel L1 0.495 validation, extreme on g00010. Source: WL/93A_…:154; WL/93C_…:120.
- [MEAS] CK2: learned local and global terms compress the albedo response, which M64 does not. Source: WL/93D_WORKLOG_ADDENDUM…:142.

**Co-adaptation: supported (WL93D), with the mechanism not isolated.**
- [MEAS] WL93D: C2_src improves sharp renders but worsens the pooled table; C2_rcv does the opposite; C2_both does not beat C0 pooled. Source: WL/93D_…FROZEN…:219-221.
- [MEAS] WL93B U0 uses untied freedom for a non-physical compensating split. Source: WL/93B_…:228-232.
- [INF] The learned receiver and global terms absorb the learned source's errors.

**Geometry specialisation: rejected for this lineage.**
- [MEAS] S1 S_geo 0.2–0.8 %; C1 swaps change nothing measurable. Floor B is the exception at S_geo 3–5 %, still far below 0.25. Source: `final93d.json`.

**Cross-cutting metric caveat.**
- [MEAS] The pooled whole-table rel L1 (broad transport) and the rendered sharp readers respond in opposite directions to source-side physical correction. WL93C and WL93D both document this. Source: WL/93C_…:194-196; WL/93D_…FROZEN…:221, 232.

## 5. Scope limits
- [FACT] Only two seeds were used. No independent sharp held-out exists (E6 was not admitted in WL93D and never generated in WL93). All render-level claims rest on inspected E2/E5. Source: WL/93D_…FROZEN…:236-238.
- [FACT] WL93D did not analyse the untied (WL93B) or dual-supervised (WL93C) operators. Source: WL/93D_…FROZEN…:240.
- [FACT] WL93 Amendment 1 is still pending owner approval. Source: WL/93_…:188.
