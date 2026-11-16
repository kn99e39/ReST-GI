# WL97 Track F0 — Independent inference-chain review (WL86–WL96A)

Auditor: independent agent, track F0. Date: 2026-11-16. Repository HEAD `efcf86a` (branch `Selective_Recompute`).
Formed before reading any other WL97 auditor's output; no file under `docs/audits/`, `results/audit97/` or
`experiments/audit97/` other than my own was opened.

## 1. Scope and method

**Question.** For each architecture-level conclusion or design constraint currently in force from WL86–WL96A:
(i) what exactly is concluded, (ii) which measurement it rests on, (iii) whether that measurement, taken at face
value, logically supports the conclusion. The verdict vocabulary is applied to the **inference**, not to a
re-measurement. Other tracks own code and numerical re-measurement.

**Method.**
- Read the verdict / interpretation / unresolved sections of WL86, 86A, 87, 92, 92A, 92B, 93, 93A–D (incl. the 93D
  checkerboard addendum), 94, 94A–D, 95, 95A, 96 (incl. `critic_physics.md`, `critic_cost.md`), 96A; the WL96A
  protocol JSON; `docs/RESEARCH_ROADMAP.md` (frontier, GM table, lines 126–158), `docs/Architecture.md`,
  `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`.
- Checked citations against the earlier worklog and, where cheap, against the stored raw JSON / NPZ with a
  numpy-only script (`experiments/audit97/F0/f0_quote_checks.py` → `results/audit97/F0/f0_quote_checks.json`).
- No GPU job, no server access, no training, no modification of any existing file.

**Labels used below.** [RAW] = recomputed by me from a stored result file. [DOC] = read in a worklog / living
document. [INF] = my inference.

**Limits of this track.** I did not read WL86B, WL88–WL91 or WL52 beyond the roadmap entries and the passages
later worklogs quote; statements about them are marked. I inspected production code only for the WL95A albedo
metric (`experiments/ecro_attr95a/a95.py:147-152, 314-319`). I did not verify that any stored number is produced
correctly by its code.

## 2. What was read and run

| item | path |
|---|---|
| Worklogs | `docs/worklogs/{86,86A,87,92,92A,92B,93,93A,93B,93C,93D(x2),94,94A,94B,94C,94D,95,95A,96,96A}_WORKLOG_*.md` (selected sections; headings listed first with `grep -n "^#"`) |
| Critics | `docs/plans/wl96_spatio_angular_carrier/critic_physics.md`, `critic_cost.md` |
| Protocol | `experiments/ecro_util96a/protocol/ecro_util96a.json` (whole) |
| Living documents | `docs/RESEARCH_ROADMAP.md`, `docs/Architecture.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md`, AGENTS.md (in context) |
| Raw results | `results/ecro86/results/final.json`, `results/gm1_explicit87/results/stage2_87.json`, `results/ecro_cf92b/results/final92b.json`, `results/ecro_attr95a/{an.json,main.json,deliv.npz}`, `results/ecro_util96a/{final,s1,cost_s1.0}.json`, `results/wl96_calc/ring_alloc.json` |
| Command | `python experiments/audit97/F0/f0_quote_checks.py` (repository root; python 3.11 + numpy) |

## 3. Headline

1. The component-level observations of the chain are, as far as an inference review can tell, real and repeatedly
   observed: the learned sharp-panel sender of the trained ECRO checkpoints is far from the physical response; a
   patch × 64-bin outgoing table biases sharp first-flight delivery independently of relation sampling; explicit
   64-bin sweeps reproduce the broad higher-order tail on CK2.
2. The **architecture-level** conclusions drawn from them are narrower than their labels. Four structural
   weaknesses recur:
   - the founding "sharp failure" of ECRO v0 is a threshold miss on **one** held-out edit that is not resolved
     against the rule statistically (56 % pass rate over state draws), while the only independently qualified
     never-inspected edit (E5) is passed by every learned model;
   - from the WL93D addendum to WL96A (nine batches) **no geometry-edit response is measured at all**; everything
     rests on CK2, a hand-iterated static scene with one grazing light at the edge of the training distribution;
   - the object under judgment drifts from "ECRO v0 as trained" to "learned sharp carriers" to "learned
     mechanisms", the last step taken in a design-only batch on scratch numbers and over a recorded dissent;
   - "utility" is judged only against an internal explicit null at N = 1,270 patches, in an overhead-dominated
     timing regime, with the learned candidate required to pay for the explicit prefix.
3. None of this rescues ECRO v0: nothing in the record shows a trained ECRO-lineage operator that is accurate on
   sharp senders, and the explicit comparator is at least as good on most measured cells. The audit finding is
   about **scope**: which rejections are earned, and which sentences exceed them.

## 4. Conclusion table (verdict applies to the inference)

| # | Conclusion in force (source) | Rests on | Inference verdict | Strongest argument that it is overdrawn | Strongest argument that it is sound |
|---|---|---|---|---|---|
| C1 | ECRO v0 GM1 viability NOT SUPPORTED; cause "learned-operator error on sharp / moderate receivers of the held-out edit" (WL86:268-272; roadmap WL86 entry; Architecture.md:13 "sharp/held-out fidelity failures") | F2 on E2 region C: 0.0717 / 0.0634 (T1), 0.0603 / 0.0588 (T0) vs 0.06, one deployed state draw, 2 seeds | **CONDITIONALLY_VALID** (valid as a frozen-gate outcome; the absolute "fails" is not statistically resolved) | [RAW] every miss is within 0.03–0.93 of the stored noise bound; over 16 independent state draws ECRO s8601 passes F1+F2 in 56 % (mean 0.0615, 95 % CI [0.055, 0.068]); E3 and E5 pass; E4 numerically passes | The gate was frozen; [RAW] the paired deficit vs explicit on identical states is real at T1 (+0.0174 ± 0.0036, 15 / 17 draws); E1 panel bias and LSPOT bias are independent symptoms |
| C2 | The learned single flight from the sharp panel under-delivers (1–18 %) and the global term compensates; "learned mapping adequacy NOT SUPPORTED" (WL86A, WL92, WL92B) | term split on E1 / E2 (WL86A); mirror-bin retention 11–14 % (WL92); counterfactual substitutions (WL92B); reproduced on E4 / E5 | **VERIFIED** as a description of the trained checkpoints (inference follows; scope "as trained, SUM / OWN contract, 2 seeds") | Cannot be separated from the sampling-limited state relation (WL86A's own caveat); H_local INCONCLUSIVE by rule; two seeds whose sign differs | Observed on four edits, two training contracts, at field and render level; later matched-norm check (WL95 §2.1) agrees |
| C3 | Explicit same-state transport ≥ ECRO; learned utility only in source-only update; the 9× per-frame gain is table structure, not learning (WL87) | 36 region-class comparisons, E1 / E2 / E3; timings | **CONDITIONALLY_VALID** | Later evidence not folded in: on E5 T1 frozen ECRO is better than explicit by 0.006–0.008 (noise bound 0.007) [RAW]; "everywhere" is cited afterwards (WL95:314, critic_cost:14,120); comparator is an internal null at one scale; Neural Full Recompute UNMEASURED | Paired, same state, same readout; the per-frame claim is a structural identity (same renderer); E5 T0 again favours explicit (+0.012…+0.019) |
| C4 | Interface information sufficient (B SUPPORTED); representation INCONCLUSIVE; "material rank-16 loses the mirror lobe" (WL92) | ENCB 0.045 / 0.052; R16 0.084 / 0.073; unconverged gate fit | **CONDITIONALLY_VALID** | Completion code 1 reads as a representation limit (wording defect disclosed); the 0.46–0.52 "floor" is in an unmatched norm (WL95 §2.2) yet is cited as a ceiling in WL93D:229 and WL94D:147,156 | The worklog itself says "not established"; no later batch treats A as decided |
| C5 | OWN supervision restores component ownership but gives no consistent held-out sharp advantage (WL92A D_STOPPED, WL92B outcome C) | field-level component errors; E4 (not admitted), E5 (admitted), 2 seeds | **CONDITIONALLY_VALID** | E5 leaves no headroom: all six models pass (max 0.037 vs 0.06); "advantage NOT SUPPORTED" is a 2-seed, one-edit comparison of differences ≈ noise bound | Reported as not established, not as refuted |
| C6 | Source-stage / untied / dual-stage supervision: STOPPED, NO_SIGNAL, SHARING_SIGNAL (WL93, 93A–C) | 128 geometries, 2,000 steps, 2 seeds; pooled metric dominated by one geometry (g00010) | **UNTESTED_INFERENCE** if read as evidence against the interventions; valid as "no signal at smoke scale" | Budget is 1 / 12 of the WL86 schedule on 1 / 8 of the geometries; the primary metric was shown insensitive to sharp readers (WL93C) | Worklogs label them feasibility only |
| C7 | REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED; geometry specialisation not supported (WL93D) | frozen OWN / T0 operators vs M64; swaps; E2 / E5 inspected | **CONDITIONALLY_VALID** | The label is a disjunction; S_geo < 1 % is forced when the operator sits at the zero level; the "physical control" M64 is itself biased for sharp delivery (WL95A) | Render-level statements are against path-traced references, not M64 |
| C8 | Reflected-albedo compression is generic and removable by training distribution; the sharp-specific residual is panel-sender delivery (WL94, 94A–C; all INCONCLUSIVE) | CK2 tables / renders; 2,000-step fine-tunes | **CONDITIONALLY_VALID** | CK2 only; `an94.W` 8 × 8 defect biases non-Lambert metrics (WL95 §2.2) and its effect on WL94-series learned numbers is unmeasured | The "dataset coverage, not representation" reading is the cautious one and is what the worklogs state |
| C9 | Sender oracle supervision: NO_RELIABLE_IMPROVEMENT (WL94D) | 1,000-step continuation, M64 teacher, CK2 evaluation | **UNTESTED_INFERENCE** if counted against a learned sender class (WL95:313 files it under "weakening") | Teacher M64 over-delivers 1.5× on CK2; 256-spp label reliability 0.53; CK2 incidence is 4.3 % of training irradiance | The worklog says "not a representation verdict" |
| C10 | ARCHITECTURE_SELECTION_PREMATURE; the learned S1 operator sits at the zero-operator level vs M64 while admissible rank-16 operators reach 0.38–0.60 (WL95) | CPU matched-norm table (§2.1) | **CONDITIONALLY_VALID** | Relative to the M64 teacher on training / validation fields; five unseparated causes (WL95's own list); "admissible operators exist" ≠ "learnable" | The matched-norm correction is the right repair of WL94D; the verdict (premature) claims little |
| C11 | P0-1 (WL95A): large learned-sender error; learned operators non-passive (albedo 2.3 / 3.1) | CK2 S1 field; max directional albedo over 42 patches × 64 bins | **CONDITIONALLY_VALID** | "Avoidable within rank 16 on this workload? Not shown" (WL95A:143); the albedo is a stored maximum for WL92A OWN checkpoints, not the WL86 deployed ones; operator is not iterated | Error is large in every norm against the physical q (rel L1 1.08–1.19) |
| C12 | P0-2 (WL95A): the binned M64 / q sender over-delivers in expectation; relation sampling does not remove it | 64 pooled relation draws; box A exactly-2-event delivery | **VERIFIED** at delivery level on CK2 (restrictions in §6) | One scene, one grazing light, one lobe; image-level relevance on CK2 unresolved (+0.0127 < 0.0167) | [RAW] 64 / 64 draws over-deliver (1.544 ± 0.035); independent workload support quoted from WL90 |
| C13 | P0-3 (WL95A): exact direction + hit-point ER evaluation closes the delivery error | pooled X1 1.022 / 0.111 | **CONDITIONALLY_VALID** | Only in the pooled expectation (64 × 512 rays per patch); the deployed draw is worse than the binned table (0.627 vs 0.524); X1 is an unbiased estimator of the reference integral, so convergence is a consistency check | [RAW] pooled L1 0.111 equals its own sampling-noise scale (≈ 0.09–0.14); the paired contrast to M64 is +0.52 ± 0.06 |
| C14 | RANK16_DELIVERY = INADEQUATE_ON_TESTED_WORKLOAD (WL95A) | bound operators fitted on the training distribution, tested at grazing CK2 | **INVALID_COMPARISON** for any capacity reading | [RAW] `ring_alloc.json`: a non-negative rank-16 factorisation is exact on the ring-0 support (max error 0); the fits gave ring 0 6.4 % of row weight | Literally true for the operators as fitted; WL96 already re-reads it as allocation |
| C15 | NO_VIABLE_LEARNED_MECHANISM, scoped, provisional (WL96) | design study; scratch numpy ports [MEAS-S]; cross-workload cost anchors | **UNTESTED_INFERENCE** | Design-only; adjudicated by the authoring session over the physics critic's EVIDENCE_INSUFFICIENT; proofs cover only linear, light-independent, separable forms; "explicit = comparator" is a canon filter, not a measurement | Scope and dissent are written down; the cost anchors quoted are reproduced correctly |
| C16 | P0-4 (WL96A): no accuracy or lifecycle headroom for a learned broad ≥ 3-event resolvent on CK2 | ROI paired change 0.00065 vs M_img 0.005; tail 3.3 ms of 144 / 88.5 ms | **CONDITIONALLY_VALID** | Static scene, one light, one N, image-level rule; field-level flag on sharp-reader tables; the candidate is forced to pay for the explicit q table and prefix; overhead-dominated timings | [RAW] decision inputs reproduce; raw dI + 2 nb is 7× below the predeclared effect |
| C17 | Design constraint in force: explicit carriers are comparators, the method must be an ECRO-lineage learned hypothesis (Architecture.md:13; roadmap:128) | human decision 2026-11-13 | not an inference (owner decision) | It acts as the admissibility filter that turns "every working design is explicit" into "no viable learned mechanism" (see F0-07) | Recorded as a decision, and WL95 / WL96 name it as owner Gate A |

## 5. Findings

Each finding: category · material · claim affected · numbers · evidence · decision-level impact · proposed verdict ·
confidence.

### F0-01 — The WL86 sharp failure is not resolved against its own rule
- **Category** statistical_weakness. **Material** yes. **Claims** P0-5; C1.
- **Numbers.** [RAW `results/ecro86/results/final.json`] E2 region C E_H and stored noise bound:
  s8601 T0 0.0603 (nb 0.0101), T1 0.0717 (0.0126); s8602 T0 0.0588 (passes), T1 0.0634 (0.0127). Margins over
  0.06 are +0.0003, +0.0117, −0.0012, +0.0034, i.e. 0.03, 0.93, −0.12, 0.27 noise bounds.
  [RAW `results/gm1_explicit87/results/stage2_87.json`, `per_edit/E2/ECRO/summary`] over 16 independent state
  realizations (ECRO s8601): T1 C mean 0.0615, SD 0.0123, 95 % CI [0.0549, 0.0680], `frac_pass_0.06` 0.5625,
  `frac_F1_F2_pass` 0.5625; T0 C mean 0.0457, pass 0.75. My recomputation including the deployed draw (17 rows):
  mean 0.0621, pass 9 / 17.
- **Reporting gap.** WL87:248 prints "ECRO 0.062, 0.012" but not the 56 % pass rate; WL87:250 gives only the
  explicit arm's 75 %. The roadmap WL86 / WL87 entries do not carry it.
- **What is solid.** [RAW] the paired ECRO − EXP difference on identical states: T1 +0.0174 ± 0.0036 (ECRO worse
  in 15 / 17), T0 +0.0054 ± 0.0039 (11 / 17). A relative sharp deficit versus explicit exists at T1 and is not
  resolved at T0.
- **Impact.** "NOT SUPPORTED under the frozen gates" stands as a frozen outcome. The narrated fact "ECRO v0 fails
  held-out sharp fidelity" — the premise of WL86A through WL96A — is, in absolute terms, a coin flip over the state
  draw on one edit. Decision-level: the size of the problem the chain set out to explain was smaller and less
  certain than the chain assumed; the relative-to-explicit deficit is the defensible statement.
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** high (numbers), medium (that no other record covers it).

### F0-02 — Held-out breadth: E2 is the only edit on which any ECRO-lineage model misses the rule
- **Category** invalid_inference. **Material** yes. **Claims** P0-5; C1, C3.
- **Numbers.** E3 (WL87:207) region C: EXP 0.0052, ECRO 0.0051 / 0.0063. E4 (WL92A, not admitted): SUM 0.025 /
  0.024, OWN 0.013 / 0.012. E5 [RAW `final92b.json`], independently qualified (Q4 SNR 13): frozen ECRO T1 C
  0.0215 / 0.0236, T0 0.0368 / 0.0295; SUM 0.028 / 0.026; OWN 0.033 / 0.027; maximum over all six models and both
  states 0.0368 against 0.06; C edit slope 0.75–1.00, corr ≥ 0.96, all PASS. Explicit control X_EXP: T1 0.0299,
  T0 0.0174. Frozen ECRO − explicit: T1 −0.0084 / −0.0062 (ECRO better; region noise bound 0.007), T0 +0.0194 /
  +0.0121 (explicit better).
- **Citation drift.** WL95:314 states "[MEAS] Explicit ≥ ECRO everywhere (WL87: 0 wins / 3 losses for ECRO)";
  `critic_cost.md:14,120` and WL96:150 repeat it (as "0 better / 3 worse / 36 equal"; WL87:216-219 says 3 worse of
  36, the rest within noise). None incorporates the WL92B E5 measurement made two days earlier.
- **Impact.** The class statement "the learned operator fails sharp held-out transport" is an E2 (+ CK2) statement.
  On the one never-inspected, independently qualified edit the frozen WL86 operator meets every WL86 gate.
- **Proposed verdict** CONDITIONALLY_VALID for C1 / C3; the word "everywhere" is INVALID as quoted. **Confidence** high.

### F0-03 — Static-scene evidence is carrying dynamic-reuse conclusions
- **Category** dataset_limitation. **Material** yes. **Claims** P0-1…P0-5; C8–C16.
- **Evidence.** Geometry-edit response measured: WL86 (E1, E2), WL87 (E1–E3), WL92A (E1, E2, E4), WL92B (E2, E4,
  E5), WL93B / 93C / 93D (inspected E2 / E5 only; E6 not admitted). Not measured at all: WL93D addendum, WL94,
  94A, 94B, 94C, 94D, 95A (`grep -c -i "edit slope|edit response|T0 / T1|STALE"` = 0 in each), and WL96A, whose
  edit is "timing-only … no quality is measured on it" (protocol `stage_3.edit`; WL96A:202). WL95 and WL96 are
  design-only.
- **CK2 provenance.** `93D_WORKLOG_ADDENDUM…:38-52`: box A placed by five render-checked iterations (p01–p05) so
  that the checker "is clearly visible"; corner x = −1.345 against the training rule's −1.35 (the boundary of the
  training distribution); one point light at grazing incidence on the panel. WL95A:120: ring-0 incidence is 4.3 %
  of training irradiance. WL94B:91: only 4.2 % of training items reach CK2's panel-mediated share
  (UNDERREPRESENTED by the predeclared rule).
- **Impact.** Every architecture-level sentence written after WL93D (rank adequacy, binned-carrier bias, learned
  resolvent utility, "no viable learned mechanism") is supported by one static geometry and one light. The
  research question — reuse under geometry edits without retraining — was last tested in WL92B. For learned-model
  arms, CK2 additionally confounds representation with training coverage.
- **Proposed verdict** UNTESTED_INFERENCE for any dynamic-reuse reading of WL94–WL96A. **Confidence** high.

### F0-04 — WL96's label states more than a design study can establish
- **Category** invalid_inference. **Material** yes. **Claims** P0-5; C15.
- **Evidence.** WL96:5-8 "design-only … nothing was implemented or trained, no renderer or GPU job"; WL96:25
  [MEAS-S] = "not evidence of record"; WL96:35,188 dissent EVIDENCE_INSUFFICIENT; `critic_physics.md:11-16,
  248-260`: "the stated proof is wrong … covers only linear, light-independent, separable forms and is
  norm-dependent … Sharp learned C and the Λ utility are unmeasured"; WL96:67-74 concedes the capacity arguments
  "prove less than stated". The verdict was adjudicated by the supervising (authoring) session between a 1–1
  critic split. The four "independent" designers received the same evidence pointers and "verified facts"
  (WL96:11), i.e. the chain's interpretations were premises, not hypotheses.
- **Object drift.** WL86 "ECRO v0 as trained" → WL95 "a geometry-conditioned learned rank-16 tied bin operator, as
  trained" → WL96 K1 "learned sharp carriers" → label "NO_VIABLE_LEARNED_MECHANISM". The text scopes it to "the
  learned mechanisms that could be specified in this study"; the label, the roadmap entry and the session memory
  index do not.
- **Placement.** The roadmap files the entry under "GM1 evidence record (measured; …)" (roadmap:130), with "Learned
  sharp carriers are excluded or dominated" stated as a result.
- **Impact.** Read at face value the label rejects a class. What the batch supports is: no *specified* candidate
  put sharp structure in a learned component, and the one learned element left was predicted to have little
  utility. The class-level exclusion of learned sharp carriers is an argument, not a tested result.
- **Proposed verdict** UNTESTED_INFERENCE. **Confidence** high.

### F0-05 — WL96A answers a narrower question than the smoke WL96 specified to settle its dissent
- **Category** invalid_inference. **Material** yes. **Claims** P0-4; C16.
- **Evidence.** WL96 §F (lines 198-236) and both critics specify: CK2 with the T0 / T1 box edit, a second
  (non-grazing) light and / or WL89 W1 or E2 T1 region C; field-level per-region accuracy ("more sensitive than
  ROI", `critic_cost.md` §7); cost at N = 1,270 / 2,802 / 5,080; an in-sample rank-ρ fit arm; confirmation only if
  "U_max < 1.25 for all N and H_acc ≤ max(2 SE, n0 / 2) in every region" (WL96:231). WL96A (protocol): CK2 'none'
  static only; decision on the image-level ROI with M_img 0.005; field level "reported, not the decision"; fit arm
  and N-scaling run only on triggers that did not fire. Under the WL96 field rule the deployed sharp-reader region
  is flagged: S1 e3_o excess 0.043 vs bound 0.024 (WL96A:119; pooled 0.016, not flagged).
- **Also.** Pre-freeze exposure: a dev figure with the real reference and real explicit operator was opened and
  "showed qualitatively that |XOD − F| … were the smallest" (WL96A:258; disclosed). Criteria were stated to be
  written before code ran; I cannot verify the ordering.
- **Impact.** WL96A:212 and the roadmap say the dissent "is now answered for the broad-last resolvent on CK2".
  That holds for one view's image-level ROI. It does not meet WL96's own confirmation rule (regions, N, workloads).
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** high.

### F0-06 — The WL96A lifecycle ceiling is conditional on who owns the prefix, and on an overhead-dominated regime
- **Category** invalid_inference. **Material** yes. **Claims** P0-4; C16; C3.
- **Numbers.** [RAW `cost_s1.0.json`] R0 edit 144.1 ms: render 59.5, geometry 30.2, q table 25.7, relation 18.1 +
  5.0, prefix 2.0, tail 3.3 (2.3 %). R0 source 88.5 ms: render 58.5, q table 24.4, prefix 2.1, tail 3.3 (3.7 %).
  U(ZERO) 1.038 / 1.043; U(LAM24) 0.93 / 1.04. GPU kernel time is 25.5 ms of the 144 ms edit (17.7 %): the
  lifecycle is CPU / framework-overhead dominated at N = 1,270.
- **Construction.** Protocol `stage_3.arms.LAM_rho`: the learned arm keeps "the same prefix (q table + first
  order)". The only learned advantage ever measured (WL87: source-only update 13–18 ms vs 51–64 ms, "SUPPORTED by
  the predeclared rule") comes precisely from *not* computing the q table. WL96A's arms exclude it by design,
  because WL96 required a physical prefix. Arithmetic only, not a measurement: a source path that needed neither
  the q table nor the sweeps would be bounded at 88.5 / (88.5 − 24.4 − 2.1 − 3.3) = 1.51× frame-inclusive
  [RAW-derived]; with WL87's 13–18 ms ECRO source update it would be roughly 1.2× (cross-batch arithmetic).
- **Scaling.** "No scaling hypothesis reaches 1.25×" (WL96A:194) is an inference from WL91's GM1 scaling table; the
  scaling extension was not run.
- **Impact.** The supported statement is: *a learned operator that replaces only the three tail orders behind an
  explicit q table and prefix cannot speed up this lifecycle at N = 1,270*. It is not a statement about learned
  transport operators that own the prefix, about larger scenes, or about optimised explicit code. The roadmap
  sentence ("behind a physical prefix") is correctly conditional; the verdict label and P0-4 are not.
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** high for the numbers, medium for the arithmetic bound.

### F0-07 — The decisive comparator is an internal explicit null whose role is fixed by decision, not by the baseline contract
- **Category** invalid_inference (with a provenance_gap). **Material** yes. **Claims** P0-4, P0-5; C3, C15–C17.
- **Evidence.** `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` ("current as of 2026-11-02") has no row for
  EXP / C3UR; their role exists only in the 2026-11-13 decision paragraph (Architecture.md:13; roadmap:128). The
  chain respects the terminology gate in wording (WL86:11, WL87:20, WL92:12, WL92A:14: no Neural Full Recompute
  claim). But the 1.25× "learned utility" rule (WL87, WL91, WL96A) and the words "decorative" (critic_cost §2.3)
  and "no viable learned mechanism" judge learning solely against explicit recomputation in a 1,270-patch room.
  Neural Full Recompute and Native Reuse are UNMEASURED across WL86–WL96A; the roadmap schedules scalability and
  external comparison at GM4.
- **Filter effect.** WL95 §5 item 1 and WL96 §E classify every design that fixes sharp delivery as "G1 class, a
  comparator under current canon". In scene86 every K is the pointwise learned ER and all remaining transport is
  free-space; under Architecture.md §2–§4 free-space state is recomputed from current geometry. "ER-exact sender +
  explicit first gather" is therefore, structurally, learned local response plus explicit free-space coupling.
  Whether that counts as the method or as a comparator is owner Gate A (WL95, WL96 say so). The "no viable learned
  mechanism" outcome is produced jointly by the evidence **and** by that classification.
- **Impact.** The explicit null is a legitimate and informative control. It does not by itself rank learned
  transport against the project's declared external baseline, and part of the WL95 / WL96 negative is a
  consequence of an admissibility rule rather than of a measurement. This audit takes no position on Gate A.
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** medium-high.

### F0-08 — Rank-16 "inadequacy" is fit-distribution allocation, and an unmatched "floor" travelled through three worklogs
- **Category** dataset_limitation / invalid_inference. **Material** yes for any capacity-based decision. **Claims** P0-1, P0-5; C4, C14.
- **Evidence.** [RAW `results/wl96_calc/ring_alloc.json`] `ring0_nonneg_rank16_exact_max_rel` 0.0; CK2 support =
  bins 12–15; row-weight share of ring 0 in the fitting norm 0.064; NMF16 ring-0 op L1 0.97 vs ring 3 0.40.
  WL95A:144 itself: "A grazing-weighted rank-16 fit was not tried". `critic_physics.md` §2.2: a light-conditioned
  rank-16 operator reaches 0.062 on CK2 [scratch]. The WL92 "Eckart–Young floor 0.46–0.52" (per-colour
  flux-Frobenius) is compared with pooled flux-L1 ≈ 1.0 in WL94D:147,156 and invoked as "the rank-16 class
  ceiling" in WL93D:229, although WL92's representation verdict is INCONCLUSIVE; WL95 §2.2 item 3 corrects the
  norm mismatch (zero operator = 1.000, SVD16 = 0.891 in the matched norm).
- **Impact.** No measurement in the chain establishes a rank-16 capacity limit for the sharp sender. The roadmap
  WL95A entry keeps the label; the correction lives in the WL96 entry.
- **Proposed verdict** INVALID_COMPARISON (capacity reading of the label). **Confidence** high.

### F0-09 — Training outcome is repeatedly close to being read as class evidence
- **Category** invalid_inference. **Material** yes. **Claims** P0-1, P0-5; C6, C9, C10.
- **Evidence.** After WL86 exactly one full-schedule retraining exists (WL92A SUM / OWN, 24,000 steps). Every later
  intervention is a 1,000–2,000-step fine-tune on 96–128 geometries with two seeds (WL93A–C, WL94, 94A, 94C,
  94D). Their frozen labels are NO_SIGNAL_AT_SMOKE_SCALE, INCONCLUSIVE (seed disagreement) and
  NO_RELIABLE_IMPROVEMENT. WL95 §2.1 lists five unseparated causes for the zero-operator level (optimisation,
  objective normalisation, generator parameterisation, sharing of `a`, tying of `c`). WL95:313 nevertheless files
  WL94D and the zero-operator level as "weakening" evidence for the learned-sender class; WL96 then drops that
  class for a different reason ("duplicates ER").
- **Impact.** "The learned S1 sender is wrong as trained" is supported. "A learned sender of this form cannot be
  trained to be right" has never been tested at scale, with a sender-directed objective, or with grazing coverage.
- **Proposed verdict** UNTESTED_INFERENCE for the class; VERIFIED "as trained". **Confidence** high.

### F0-10 — Confirmation: the binned-carrier delivery bias and its removal by exact evaluation (CK2, delivery level)
- **Category** confirmation. **Material** yes (supports P0-2, P0-3). **Claims** C12, C13.
- **Numbers.** [RAW `results/ecro_attr95a/deliv.npz`, box A = patches 1130–1183, reference = mean of `ref_a`,
  `ref_b`] M64 (A4): draw ratio mean 1.544, SD 0.283, SE 0.035, min 1.078, **64 / 64 draws > 1**; deployed 1.208 at
  the 14th percentile of the draws (a plausible low draw, as WL95A says). Physical q (q256): 1.422 ± 0.034, 63 / 64
  > 1. X1: 1.022 ± 0.050 (median 0.92: skewed), pooled rel L1 0.111 against a sampling-noise scale of ≈ 0.09
  (mean-absolute-deviation estimate; ≈ 0.14 with a per-patch standard-error estimate, the draws being heavy-tailed)
  and n0 = 0.059; M64 pooled rel L1 0.546 against noise scale ≈ 0.09 by either estimate (bias-dominated). Paired M64 − X1 ratio:
  +0.52 ± 0.06, positive in 55 / 64 draws. All reported WL95A B2 values reproduce (1.544 / 0.546, 1.422 / 0.433,
  1.022 / 0.111, 1.037 / 0.268; `an.json` Ra 0.232 ± 0.047, Rp 0.157 ± 0.060).
- **Impact.** The inference "the bias sits in the binned outgoing table, not in relation sampling" follows from the
  data on this configuration.
- **Proposed verdict** VERIFIED (scope: CK2, box A, exactly-2-event S1-mediated delivery). **Confidence** high.

### F0-11 — P0-3 is true in expectation only, and is not shown to matter in the CK2 image
- **Category** invalid_inference (scope). **Material** moderate. **Claims** P0-3; C13.
- **Numbers.** [RAW] single deployed draw: X1 rel L1 0.627 vs M64 0.524; per-draw median 0.617 vs 0.726. ROI
  (WL95A B3): pooled X1 0.0124, floor 0.0118, pooled M64 0.0245; the +0.0127 is below the 0.0167 tolerance
  (WL95A:135; WL96:69). X1 evaluates the reference integrand exactly at sampled rays, so its pooled convergence is
  a consistency check of the contracts rather than a property of a representation.
- **Impact.** "Substantially closes" needs "pooled over 64 relation draws, at the delivery level". As an estimator
  at the deployed budget it does not close the error; at image level on CK2 neither the bias nor its removal is
  resolved.
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** high.

### F0-12 — The passivity statement rests on a stored maximum for a different checkpoint family
- **Category** provenance_gap / statistical_weakness. **Material** moderate. **Claims** P0-1; C11.
- **Evidence.** `a95.py:147-152, 314-319`: `albedo_max` = maximum over incident bins of Σ_out A·W_T, then maximum
  over the 42 S1 patches; `main.json` stores only that maximum (2.264 / 3.077) and `passive_all`. The same record
  gives `mass_mean` 0.869 / 0.835 relative to M64 (whose mean albedo is 0.635): the learned operators are
  sub-unit on average. The operators are WL92A OWN s8601 / s8602, not the WL86 deployed `ecro_s860x`. No worklog
  measures passivity of the WL86 checkpoints (WL86:41: "No single-scattering passivity is imposed"). ECRO applies
  the local operator in a fixed-depth composition, not an iterated solve; WL86 reports aggregate flux 0.98–1.00.
- **Not recorded.** How many (patch, bin) rows exceed 1, at which incidence, and what share of deployed irradiance
  they carry.
- **Impact.** "OWN S1 local operators exceed unit directional albedo for some incident bins" is supported.
  "The deployed learned operator violates passivity" is not measured; the hedged P0 wording ("may") is the most
  the record allows.
- **Proposed verdict** CONDITIONALLY_VALID (the deployed-operator reading is UNTESTED_INFERENCE). **Confidence** medium.

### F0-13 — Chain premises: later batches inherit an interpretation or a biased control from earlier ones
- **Category** invalid_inference. **Material** moderate. **Claims** P0-1, P0-5; C7, C9.
- **Instances.**
  1. M64 is the "physical control" in WL93D and the teacher in WL94D; WL95A shows the M64-binned sender itself
     over-delivers 1.5× on CK2 (and WL95 §2.2 item 5 retracts WL94D's "floor exists even for the exact sender" as
     overstated). Render-level conclusions survive (path-traced references); operator-level "80 % wrong vs
     physics" in WL93D is "vs M64".
  2. WL92's unmatched rank-16 "floor" → WL93D:229 → WL94D:147 (F0-08).
  3. WL87's "explicit ≥ ECRO" → WL95:314 → critic_cost:14 → WL96 §D conclusion, without E5 (F0-02).
  4. WL95A's delivery-level bias → all four WL96 designs' premise, although WL96:69 records that image-level
     relevance on CK2 "is not demonstrated".
  5. `an94.W` (8 × 8 instead of 4 × 16; WL95 §2.2 item 1) entered every non-Lambert exitance metric of WL94–94D;
     "its effect on the learned and render-derived WL94-series numbers is unmeasured". The labels of those batches
     are INCONCLUSIVE / NO_RELIABLE_IMPROVEMENT, so no positive or negative decision hangs on it, but the
     quantitative statements (e.g. energy 1.71–1.75) are unverified. Left to the measurement tracks.
- **Proposed verdict** CONDITIONALLY_VALID overall; item 5 is an open IMPLEMENTATION_DEFECT of unmeasured impact.
  **Confidence** medium.

### F0-14 — Two seeds throughout; seed effects exceed the effects being attributed
- **Category** statistical_weakness. **Material** moderate. **Claims** P0-1, P0-5.
- **Evidence.** Seed-to-seed operator difference 26–32 % vs geometry drift 4–14 % (WL93D); learned sender flux
  0.52× (s8601) vs 1.63× (s8602) on CK2 (WL95A B2); OWN E2 T1 0.077 vs 0.040 (WL92A); WL94A / WL94C INCONCLUSIVE
  by seed disagreement; WL95A NEXT_DESIGN_TARGET differs by seed.
- **Impact.** Signed and magnitude statements about "the learned sender" are statements about two checkpoints.
- **Proposed verdict** CONDITIONALLY_VALID. **Confidence** high.

### F0-15 — Quote checks (confirmations and minor misquotes)
- **Category** confirmation (plus non-material citation errors). **Material** no.
- **Reproduced exactly from raw files.** WL86 E2 C values; WL87 EXP realization mean / SD / range (0.044, 0.017–0.018,
  0.024–0.077; 75–76 %); WL92B E5 table and controls; WL95A B1 operator metrics, B2 delivery, flight attribution;
  WL96A decision inputs (H −0.00082, U −0.00077, dI(XOD) 0.00065, n_img 0.00146, tol_E 0.0163), ROI table, U
  table, R0 breakdown; WL96's Amdahl arithmetic from WL91:228-246 (124.7 / (124.7 − 10.6) = 1.09; 75.8 / 65.2 =
  1.16); WL96 ring-0 statement.
- **Minor.** (a) "0 better / 3 worse / 36 equal" (WL96:150, critic_cost:120) should be 33 within noise of 36.
  (b) WL96A:178 / roadmap "9–40× further from the floor": [RAW] G / XOD = 39× (s8601), 20× (s8602); G / XD = 8.8×,
  4.3×; the quoted range mixes comparators and fits seed 8601 only. (c) `final92b.json` `outcome.supported` lists
  only H_geometry while the worklog table prints H_interact SUPPORTED (reconciled in WL93). (d) WL87's roadmap
  sentence "is never worse than ECRO beyond noise (worse in 3 sharp classes of 36)" is ambiguous about which arm
  is worse.

### F0-16 — Living-document statements that exceed the tested scope
- **Category** invalid_inference. **Material** yes (these sentences steer the next decision). **Claims** P0-5.
- See §8 for the list. **Proposed verdict** per item there. **Confidence** medium-high.

## 6. Priority-0 claims within this lens

| Claim | Verdict on the inference | Why | Restrictions that must travel with it |
|---|---|---|---|
| **P0-1** learned S1 sender has large avoidable error; deployed operator may violate passivity | **CONDITIONALLY_VALID** | "Large error": supported on every geometry and norm (C2, C10, C11). "Avoidable": shown only as existence of better admissible operators against the M64 teacher on training / validation fields (WL95 §2.1); on CK2 the rank-16 bounds are as wrong (WL95A:143); never shown learnable. "May violate passivity": a stored maximum for WL92A OWN checkpoints (F0-12) | as trained; 2 seeds; M64-relative for "avoidable"; passivity measured on OWN S1 local factors only, consequence for deployed transport unmeasured |
| **P0-2** the M64 / 64-bin sender introduces systematic sharp-delivery bias not eliminated by relation sampling | **VERIFIED** (delivery level) | [RAW] 64 / 64 draws over-deliver, 1.544 ± 0.035; q256 1.422 ± 0.034; independent support quoted from WL90 on the GM1 workload (not re-read in depth) | CK2 box A, one grazing light, S1 lobe, patch × 64-bin piecewise-constant outgoing table; magnitude is configuration-dependent (WL96 scratch: 0.05–0.22 for other lights); image-level relevance on CK2 unresolved |
| **P0-3** continuous exact-direction and hit-point ER evaluation substantially closes the CK2 delivery error | **CONDITIONALLY_VALID** | True for the pooled expectation (0.546 → 0.111, at noise); false for the deployed single draw (0.627 vs 0.524); not resolved at image level (F0-11) | pooled over 64 relation draws; delivery-level metric; trades bias for heavy-tailed variance |
| **P0-4** the learned broad higher-order resolvent has no meaningful accuracy or lifecycle headroom over explicit recurrence on CK2 | **CONDITIONALLY_VALID** | Accuracy: image-level ROI paired change 0.00065 vs 0.005 (sound for that ROI). Cost: 1.04× ceiling for a resolvent replacing only the tail orders (F0-05, F0-06) | static CK2, one light, one view, N = 1,270; oracle path-traced prefix; learned arm pays the explicit q table / prefix; field-level flag on sharp-reader tables (0.043 vs 0.024, deployed); overhead-dominated timing; no scaling; no edit quality; trainability and generalisation untested |
| **P0-5** the negative results justify rejecting the tested ECRO representation mechanisms, but do not establish a general impossibility for reusable learned light transport | **CONDITIONALLY_VALID** (second clause VERIFIED; first clause valid only for the items in §7.1) | See §7 | "tested mechanisms" must be read as trained instances and one carrier, not as representation classes |

## 7. Assessment of P0-5

### 7.1 Legitimately rejected, and at what scope

| Tested thing | What the evidence earns | Scope |
|---|---|---|
| ECRO v0 checkpoints (WL86, SUM contract) as the GM1 method | Not adopted: no fidelity advantage over the explicit same-state table, a paired sharp deficit on E2 T1 and E1 panel, cone-source bias, and edit-to-frame cost within 15 %. The absolute gate failure is marginal (F0-01) | as trained, 2 seeds, scene86 room family, rigid box edits, point / cone lights |
| The learned tied rank-16 local operator **as the owner of the S1 sharp first-order response, as trained** (SUM, OWN and the smoke continuations) | Inaccurate: zero-operator level vs M64, 1–18 % / 15–51 % family delivery, seed-signed | as trained; not the rank-16 class (F0-08), not learnability (F0-09) |
| OWN component supervision as a sufficient fix | Not established: ownership recovers, held-out sharp advantage not consistent | E5, 2 seeds; not refuted |
| A patch × 64-bin piecewise-constant outgoing table as the carrier of sharp once-scattered radiance (learned or explicit) | Biased in expectation for the S1 lobe; sampling does not remove it | CK2 delivery level [RAW]; GM1 image level per WL90 [DOC]; not other bin counts, bases or anchors |
| A learned broad ≥ 3-event resolvent placed behind an explicit q table and prefix | No image-level accuracy headroom and ≤ 1.04× lifecycle headroom | CK2 static, N = 1,270, this implementation |

### 7.2 Not legitimately rejected (verdict labels notwithstanding)

- **The rank-16 / low-rank bin-operator class** for the sharp sender: WL92 A INCONCLUSIVE; WL95 admissible rank-16
  operators at 0.36–0.54; exact on ring-0 support; the binding issue named by the physics critic is
  light-independence over the hemisphere, untested.
- **Source-stage, untied, dual-stage, material-mix, query-weighted and sender-teacher supervision** (WL93A–C,
  WL94–94D): smoke-scale, two seeds, frozen labels INCONCLUSIVE / NO_SIGNAL. They are not negative results.
- **Learned sharp carriers as a class** and **"learned mechanisms"** (WL96): design-only inference with a recorded
  dissent; sharp learned free-space transport untested (the WL52 prior is for broad Lambertian senders).
- **The learned resolvent at other scales, other workloads, or owning the prefix**: untested.
- **ECRO-lineage dynamic reuse**: the edit-response gates (F3, C-region slope on E5) are passed by the trained
  operators wherever they were measured after WL86; no batch after WL92B measures an edit.

### 7.3 "No general impossibility" — why this clause is safe

One room family; surface-only scenes in which every K is the pointwise ER (the FNTO learned object, nonlocal
intrinsic K, is not exercised); point / cone lights; rigid box edits; two training seeds; 1,270 patches; an
internal explicit null as the only comparator; Neural Full Recompute, Native Reuse, larger scenes, non-rigid
edits and mixed materials unmeasured; the last nine batches static. The record cannot support an impossibility
statement, and the worklogs do not make one explicitly. The risk is in labels and summaries (§8), not in the
bodies of the worklogs.

## 8. Living-document statements that exceed the tested scope

| Where | Statement | Why it exceeds | Verdict |
|---|---|---|---|
| `docs/Architecture.md:13` | ECRO v0's "sharp/held-out fidelity failures" (plural, general) | One region class on one of four held-out edits, within one noise bound of the rule, 56 % pass over state draws; E3 / E5 pass | CONDITIONALLY_VALID |
| `docs/RESEARCH_ROADMAP.md` WL86 / WL87 entries | "misses the region rule on the held-out edit's sharp / moderate receivers"; explicit "never worse than ECRO beyond noise" | The ECRO pass rate over state draws (stored in WL87's JSON) is absent; E5 T1 later shows the opposite sign at the noise bound | CONDITIONALLY_VALID |
| Roadmap WL92 entry | "a material-level rank-16 bin factorisation loses the sharp panel's mirror lobe (flux Eckart–Young residual 0.46–0.52 …)" | True for that construction; the residual is in a norm later shown unmatched to the quantities it was compared with; reads as a capacity statement | CONDITIONALLY_VALID |
| Roadmap WL95A entry | "RANK16_DELIVERY INADEQUATE_ON_TESTED_WORKLOAD" | Fit-distribution allocation, not capacity (corrected only inside the WL96 entry) | INVALID_COMPARISON (capacity reading) |
| Roadmap WL96 entry, under "GM1 evidence record (measured …)" | "NO_VIABLE_LEARNED_MECHANISM … Learned sharp carriers are excluded or dominated"; scope includes "rigid edits" | Design-only; scratch numbers; dissent; no edit was evaluated in the underlying CK2 evidence | UNTESTED_INFERENCE |
| Roadmap WL96A entry | WL96's "dissent is answered for the broad-last resolvent on CK2"; "9–40× further from the floor" | Narrower test than WL96 specified (F0-05); range fits one seed and mixes comparators | CONDITIONALLY_VALID |
| Roadmap WL96A entry | "Dated constraint … on CK2 there is neither accuracy nor lifecycle headroom for a learned ≥ 3-event resolvent behind a physical prefix" | Correctly conditional as worded; "accuracy" holds at the image-level ROI only | CONDITIONALLY_VALID (wording acceptable) |
| `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` | (absence) no role entry for the explicit same-state comparator that decides WL87–WL96A utility | provenance gap, not an over-statement | — |
| Session auto-memory index (outside the repository; primes future sessions) | "explicit same-state transport >= ECRO everywhere"; "sharp fix always explicit; learned resolvent decorative"; "rank-16 INADEQUATE on CK2 grazing" | Compressed labels without scope; the first is contradicted at E5 T1 | exceeds scope |

AGENTS.md carries no WL86–WL96A verdict text; nothing to flag there.

## 9. Special checks requested by the brief

| Check | Result | Evidence |
|---|---|---|
| Over-generalisation from one workload | **Present** in labels and summaries; bodies mostly scope correctly | F0-03, F0-04, §8 |
| Training failure confused with capacity failure | **Partly present**: WL95:313 "weakening", RANK16 label; WL92 code 1 (disclosed) | F0-08, F0-09 |
| Implementation defect confused with architecture failure | Not found as a driver of a verdict within this lens; `an94.W` and `delivery_box` defects were found by the chain itself (WL95 §2.2) and touch only INCONCLUSIVE batches; impact unmeasured | F0-13 item 5 |
| Dataset coverage confused with representational impossibility | **Present** for the rank-16 label; correctly avoided for material compression (WL94) | F0-08; C8 |
| Static behaviour confused with dynamic-reuse evidence | **Present**: zero edit-response measurements in WL94–WL95A; WL96A edit timing-only | F0-03 |
| Explicit comparator used to judge neural superiority without a valid baseline role | Terminology respected; **utility judged only against an internal null**, role absent from the baseline contract | F0-07 |
| Silent change of the object under test | **Present**: "ECRO v0 as trained" → "learned sharp carriers" → "learned mechanisms"; "deployed operator" (P0-1) vs measured OWN checkpoints | F0-04, F0-12 |
| Frozen-mapping labels whose INCONCLUSIVE outcomes are later narrated as negative | **Partly present**: WL94D → "weakening" (WL95:313); WL92 "floor" as ceiling; WL86 "GENERALIZATION NOT SUPPORTED" although "field-level evidence does not show memorisation" (WL86:268) | F0-08, F0-09 |
| Batch N+1 taking batch N's interpretation as a premise | **Present** in five places | F0-13 |
| Citation checks | 20+ quoted numbers reproduce from raw; four minor misquotes | F0-15 |

## 10. What I could not verify, and why

- Whether any stored number is correctly produced by its code (out of this track's lens; no server access used).
- The fraction and location of non-passive (patch, bin) rows of the learned operators, and passivity of the WL86
  checkpoints: only a maximum is stored locally; needs the checkpoints.
- The effect of the `an94.W` 8 × 8 weighting on WL94–94D learned and render-derived numbers.
- That WL96A's criteria were written before any WL96A code ran, and that the disclosed dev-figure exposure did not
  influence the choice of an image-level primary rule.
- WL90 / WL91 (independent support for P0-2 on the GM1 workload) and WL52 (prior used against learned C): quoted
  from roadmap entries and later worklogs only.
- Image-level relevance of the sharp-prefix bias on CK2: unresolved in the record itself (2 × 4,096 spp tolerance).
- Whether ECRO seed 8602 shows the same 56 % state-draw pass rate: `stage2_87.json` holds one ECRO seed (first
  row 0.0717 = s8601).

## 11. Reproducibility appendix

```
# repository root, python 3.11 + numpy only, no GPU, no network
python experiments/audit97/F0/f0_quote_checks.py
# -> results/audit97/F0/f0_quote_checks.json  (inputs' sha256 prefixes are stored under "inputs")
```

Read-only inspection commands used (Git Bash):
```
grep -n "^#" docs/worklogs/<NN>_WORKLOG_*.md              # section maps
sed -n '<a>,<b>p' docs/worklogs/<file>                    # verdict / interpretation sections
grep -c -i -E "edit slope|edit response|T0 ?/ ?T1|T0↔T1|signed change|STALE" docs/worklogs/{94_,94A,94B,94C,94D,95A,96A}*.md
grep -l -i "neural full recompute" experiments/*/protocol/*.json
```

Key raw fields:
- `results/ecro86/results/final.json` → `seeds/<seed>/fidelity/E2/<T>/C/{E_H,noise_bound}`, `STATUS`.
- `results/gm1_explicit87/results/stage2_87.json` → `per_edit/E2/{EXP,ECRO}/rows[*]/T{0,1}/C/E_H` (17 rows,
  seeds 86001, 87201–87216), `per_edit/E2/ECRO/summary/{T1:C:E_H, T1:C:frac_pass_0.06, frac_F1_F2_pass}`.
- `results/ecro_cf92b/results/final92b.json` → `e5/models/*/fidelity/T{0,1}/C/E_H`, `e5/controls/X_EXP_T{0,1}`,
  `e5/criteria`, `hypotheses`, `outcome`.
- `results/ecro_attr95a/deliv.npz` → `draws|<arm>` (64 × 1270), `dep|<arm>`, `ref_a`, `ref_b`; box A = 1130:1184.
- `results/ecro_attr95a/main.json` → `operator/*/{albedo_max,mass_mean,passive_all}`.
- `results/ecro_util96a/s1.json` → `decision`, `images/<arm>|ROI_T`; `cost_s1.0.json` → `arms/R0/{edit,source}`, `U`.
- `results/wl96_calc/ring_alloc.json`.

Resumption check (2026-11-16, same track, after the first attempt was interrupted at reporting): the report was
re-read for completeness and the decisive numbers were recomputed with fresh inline numpy code, independent of
`f0_quote_checks.py`, directly from the raw files: WL87 E2 T1 C state-draw statistics (stored summary 0.0615, CI
[0.0549, 0.0680], `frac_pass_0.06` 0.5625; 17-row mean 0.0621, 9 / 17; paired ECRO − EXP +0.0174 ± 0.0036, 15 / 17;
T0 +0.0054 ± 0.0039, 11 / 17), WL95A box-A delivery (M64 / A4 1.544 ± 0.035, 64 / 64 > 1, pooled 0.546; q256
1.422 ± 0.034, 63 / 64; X1 1.022 ± 0.050, pooled 0.111; deployed X1 0.627 vs A4 0.524), WL92B E5 (max C E_H 0.0368;
frozen ECRO − X_EXP T1 −0.0084 / −0.0062, T0 +0.0194 / +0.0121), WL96A decision and U values. All agree with the
report. One refinement was made: the pooled-X1 noise scale is estimator-dependent (0.09 vs 0.14), now stated in
F0-10 and row C13; no verdict changes. Quoted document lines WL95:314, WL96:231, WL96A:119 / 178 / 194 / 212 / 258,
Architecture.md:13 and roadmap:130 / 155 were re-opened and match.

Seeds: none drawn by this audit (deterministic recomputation of stored arrays).
Files created by this track: this report; `experiments/audit97/F0/f0_quote_checks.py`;
`results/audit97/F0/f0_quote_checks.json`. No existing file was modified; nothing was committed.
