# WL95 read-only design input: canonical objectives, ownership contracts, rejected families and constraints

Scope: I read docs/Architecture.md, RESEARCH_CENTRIC_TOPIC.md, BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md, FNTO_ARCHITECTURE_CONTRACT_V0_1.md, README.md, all of RESEARCH_ROADMAP.md (167 lines), and AGENTS.md. I skimmed docs/plans/global_transport_study_86b/ (dossier, panels), docs/plans/gm1_architecture_review/ (README, 00, 60), docs/plans/wl80a/final_architecture_decision.md and coordinator_disposition.md, and worklogs 80A, 80B, 84A and 86B. No file was edited and nothing was run.

## (A) Canonical objectives

**Scientific objective**
- [FACT] The adopted objective is to keep GI physically credible and high quality after geometry edits. Transport that stays valid is reused, and invalid transport is updated at a favourable quality-matched end-to-end cost. Compressing K is a means to that end, not the goal (RESEARCH_CENTRIC_TOPIC.md:9; Architecture.md:7).
- [FACT] The intended contribution has to show three properties *jointly*:
  1. complex transport fidelity: sharp glossy/specular and multi-bounce;
  2. dynamic edit response, including stationary receivers;
  3. quality-matched economics.
  A BSDF surrogate, a K/C partition, a cache or a selective update alone does not count (RESEARCH_CENTRIC_TOPIC.md:11; roadmap RESEARCH_ROADMAP.md:21(i)).
- [FACT] Central hypothesis (INCONCLUSIVE): a source-independent reusable intrinsic rule can be separated from geometry-dependent within-domain state, and selective state rebuild saves cost without losing fidelity. It must earn its benefit beyond exact optical-domain selective recompute (RESEARCH_CENTRIC_TOPIC.md:13).
- [FACT] Geometry-dirty and transport-dirty sets differ: stationary receivers can change (RESEARCH_CENTRIC_TOPIC.md:7; AGENTS.md:125).

**Geometry-change reuse without retraining**
- [FACT] Rule parameters are "fixed across the supported edits; no post-edit retraining". Geometry-dependent intrinsic state is kept only with a valid dependency/correspondence certificate. Cached coefficients are invalidated with their actual dependencies. Free-space state updates for affected air paths. Radiance is recomputed through the current coupled solve (Architecture.md:48-54).
- [FACT] Reusing network weights is not the same as preserving computed state. A source-swap test is not evidence of selective geometry-state reuse (Architecture.md:56).
- [FACT] FNTO v0.1 sets "0 updated learned parameters after a rigid edit" as the target, not as an achievement (FNTO_ARCHITECTURE_CONTRACT_V0_1.md:103-107).
- [FACT] Rigid motion preserves intrinsic state only with consistent material-frame correspondence. Non-rigid validity cannot be inferred from locality or fixed weights. Missing validity information means conservative invalidation or a full rebuild (Architecture.md:62-64).
- [FACT] Long-term coverage is mandatory: rigid motion, local deformation, global bending/twisting and self-contact (Architecture.md:66; RESEARCH_CENTRIC_TOPIC.md:20-21).
- [FACT] "Selective" has two coexisting definitions that must not be merged:
  - AGENTS.md:107-109: complete owned *learned* state of each invalidated Surface-Owned unit, which involves retraining;
  - Architecture.md:70: same-rule F/S *state* rebuild, no retraining.
  Architecture.md:76-78 keeps the Surface-Owned contract as a separate baseline.

**K/C ownership**
- [FACT] An optical domain is a material/medium region or a surface interface, not an authoring object (Architecture.md:17).
- [FACT] K owns: boundary incidence → response at the first exit into free space, local response, in-medium transport, internal scattering and transmission (Architecture.md:19).
- [FACT] C owns: an outgoing event through editable free space to the next entry, including same-asset and same-domain re-entry (Architecture.md:20).
- [FACT] Every segment has exactly one owner. Object identity must not freeze exposed interreflection into K (Architecture.md:25; FNTO_ARCHITECTURE_CONTRACT_V0_1.md:39-40, 48-50).
- [FACT] Direct and C-mediated incidence use the same source-independent rule. No source-identity branch and no direct/indirect branch is allowed (Architecture.md:21).
- [FACT] Solve form: `i_a = e_a + Σ_b C_ab(G) b_b`, `b_a = K_a[i_a]`. K must be linear, nonnegative and passive at fixed geometry. Ballistic and scattered light are separate families, each read out exactly once. Repeated composition must be stable. Near-contact spatial information must survive both sides of the interface (Architecture.md:31-38).
- [FACT] The physical reference is the path-history-independent homogeneous RTE volume, not the object-scoped Burley target (Architecture.md:23; BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:48).

**Neural Full Recompute terminology (binding)**
- [FACT] Neural Full Recompute for 8DNA/RNA means native T1 retraining/rebuild/refit of the *learned representation* when it is actually invalidated. It never means a Cycles/Mitsuba re-render (AGENTS.md:13; BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:9, 23).
- [FACT] Four separate roles: Neural Full Recompute, Native Reuse (T0 asset still valid), internal same-checkpoint F/S, and path-traced re-render (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:25-28; Architecture.md:70; RESEARCH_CENTRIC_TOPIC.md:22).
- [FACT] Mandatory gate before any comparative prompt:
  - name the owned learned state of each arm;
  - show that T0→T1 actually invalidates it;
  - declare reuse, cold rebuild, refit or F/S;
  - fix a common workload and T1 quality;
  - declare the timing boundaries;
  - keep an independent path-traced ground truth.
  If this cannot be done, report UNMEASURED (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:30).
- [FACT] WL69's CY arm was a Cycles re-render, and WL70's 11.8× was internal sun-state reuse. External Neural Full Recompute superiority is WITHDRAWN/UNMEASURED (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:32).

**Evaluation requirements**
- [FACT] GM1 requires all of the following, with no T1 supervision or oracle at runtime (RESEARCH_ROADMAP.md:16):
  1. REF-P, the layered physical path-traced reference;
  2. REF-ER, full path tracing with the same frozen ER;
  3. one explicit proposed mechanism;
  4. stale/no-update and same-mechanism full rebuild;
  5. directional/highlight, signed-change, multi-bounce and visibility-affected/unaffected evidence;
  6. all update/solve/readout/memory costs, with predeclared quality-matched edit-to-valid-output and per-frame accounting.
- [FACT] Local ER error must be separated from global/composition error (RESEARCH_ROADMAP.md:21(ii); Architecture.md:44).
- [FACT] Predeclare scenes, splits, seeds, precision and thresholds. Keep held-out edits out of fitting. Isolate the declared mechanism. Test unseen spatial/directional incident fields and all-boundary sender outputs. Report absolute quality separately from F/S equivalence and signed change. Use a valid spectral-radius estimator (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:54-58, 62).
- [FACT] One asset, material family or edit cannot establish generality (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:70).

## (B) Current working architecture and status

- [FACT] **FNTO v0.1** is the "unchanged historical design contract dated 2026-10-23". Its "not implemented" status applies to that date only (Architecture.md:9; README.md:11). The v0.1/r1 recipes are NOT SUPPORTED (Architecture.md:90).
- [FACT] The K/C ownership contract is retained. The representation and update mechanism beyond ER are open (Architecture.md:11).
- [FACT] **Local layer:** WL84 learned ER is the primary local-scattering development baseline (human decision, 2026-11-10). It is component level only, qualified on one coated-conductor family, and "not a spatially nonlocal operator". Learned E is kept as a control (Architecture.md:11, 42, 86; RESEARCH_ROADMAP.md:15).
- [FACT] **Global layer:** ECRO is resumed as the "central implementation base for GM1" by human decision on 2026-11-13. This is "not acceptance of the WL86 frozen ECRO v0 result". Sharp/held-out fidelity failures, learned-flight under-transfer and compensation remain unresolved. EXP/C3UR stay learning-free comparators/diagnostics, "not the global method to be published" (Architecture.md:13; RESEARCH_CENTRIC_TOPIC.md:29; RESEARCH_ROADMAP.md:128).
- [FACT] What the next decision must do: isolate one falsifiable ECRO-lineage representation/training/ownership hypothesis that addresses the sharp learned-flight deficit and compensation, without losing source-independent reuse, edit response or economics (RESEARCH_ROADMAP.md:128).
- [FACT] ECRO v0 form:
  - physical transport for ≤2 events;
  - a geometry-conditioned, source-independent linear operator that fills a patch × 64-bin table of ≥3-event radiance (RESEARCH_ROADMAP.md:133);
  - internally, tied local factors a·cᵀ, a single-flight term of exactly 2 events and a global term of 3-5 events, supervised on the sum only (RESEARCH_ROADMAP.md:140).
- [MEAS] WL86: NOT SUPPORTED under frozen gates. E_H on the held-out sharp/moderate region is 0.063/0.072 against a 0.06 gate. Economics NOT TESTED; descriptively 3-6× (RESEARCH_ROADMAP.md:133).
- [MEAS] WL87: explicit same-state transport is never worse than ECRO beyond noise. The only measured learned advantage is a 3.5-3.9× cheaper source-only update (RESEARCH_ROADMAP.md:135).
- [MEAS] WL92-WL94D suffixes: no adopted fix. Recurring findings:
  - the learned panel once-scattered sender is the residual (RESEARCH_ROADMAP.md:142, 151-152);
  - sum/global co-adaptation and compensation (RESEARCH_ROADMAP.md:142, 147);
  - seed disagreement (RESEARCH_ROADMAP.md:149, 151);
  - the rank-16 bin-factorisation floor 0.46-0.52 was not approached (RESEARCH_ROADMAP.md:152).
- [FACT] Architecture §8's disposition table has no ECRO, EXP or C3UR row. ECRO appears only in the §1 decision text (Architecture.md:13 vs 84-101).
- [FACT] ECRO partitions transport by event count, not by K/C segment, and no canonical document places it relative to the K/C contract. This was flagged in the WL86B dossier (global_transport_study_86b/dossier.md:448; 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:48) and is still unresolved in the current Architecture text [INF from reading Architecture.md:13-38].
- [FACT] The open design question is GM1: which global spatial/angular representation and edit-update mechanism preserves sharp glossy and multi-bounce response under a rigid edit with frozen ER. "One mechanism must be explicitly chosen before its experiment" (Architecture.md:107).

## (C) Previously considered and rejected or unadopted families

| Family | Disposition | Stated reason | Source |
|---|---|---|---|
| Intra-model feature masking (8DNA triplane) | Rejected as main ownership route | A jointly owned shared decoder owns transport. It is not a test of complete-unit ownership | [FACT] Architecture.md:88; AGENTS.md:109 |
| SPT-1 (free-space relation state + persistent heads) | NOT SUPPORTED; no SPT-1+SSS patch | [MEAS] −11.3 dB static. The free-space-only, low-bandwidth state cannot express object-internal SSS or transmission | Architecture.md:89; AGENTS.md:152 |
| FNTO v0.1 four-part learned K (WL41) | NOT SUPPORTED | [MEAS] 64/64 solves diverge. Nonlocal kernels are non-passive outside the probe span | AGENTS.md:44; Architecture.md:90 |
| FNTO r1 (WL42) | NOT SUPPORTED | [MEAS] Static −9.08 dB vs RNA. Passive only under a hinge (ρ 0.68 vs 10.47); intrinsic fidelity limit | AGENTS.md:48 |
| K_k44P | Historical recipe, not current | Directional-sender and reference-semantics limits. It is non-passive as a directional sender (ρ 9.4/17.1) | [FACT] Architecture.md:91; AGENTS.md:52 (still calls it "current", which is stale) |
| DASQ / L0C / L1C (WL49-50) | Not adopted | [MEAS] A first-event gain does not survive composition: ρ 6.9-9.0, every coupled solve diverges. Cause per WL51: recursion of a vertex-specific Burley reference operator. Not DASQ-specific | Architecture.md:93; AGENTS.md:76, 80 |
| Learned inter-domain C / GLT (WL52) | NOT SUPPORTED | [MEAS] Valid and stable (ρ 0.51), but update is 8-19× slower than GPU explicit C. Fails contact/visibility-transition locality. Dense O(N·M), so no asymptotic gain | Architecture.md:94; AGENTS.md:84 |
| CAK, contact-adaptive co-refined K/C cells (WL55) | Not adopted | Misses its complete accuracy contract (oblique groups PARTIAL) and the tractability rule. The spatial-resolution requirement is kept | [FACT] Architecture.md:95; [MEAS] AGENTS.md:96 |
| Exact domain-selective oracle (WL56) | Baseline only | Full/selective equivalence ≠ physical GI quality ≠ learned speedup. [MEAS] Fails 9/12 | Architecture.md:96; BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:12 |
| Learned D / Rule–State R (WL57) | NOT SUPPORTED for integration | Dynamic/directional fidelity fails. Superiority INCONCLUSIVE | Architecture.md:97 |
| intrinsic_state_reuse_v1 (WL58) | Not adopted | Finite angular interface fails the sharp-GGX probe | Architecture.md:98 |
| persistent_intrinsic_FS_v1 (WL59) | Tested, not adopted | F≡S computationally, but both fail fidelity. No quality-matched cost benefit | Architecture.md:99 |
| Fixed-family contrast supervision (WL64) | NOT SUPPORTED | Held-out directional quality worsens | Architecture.md:100 |
| BGC-Transport (WL80A Track A) | Rejected | Out-of-orbit contexts not identified; targets a secondary factor; re-assembles tested mechanisms; wrong normalisation | [FACT] wl80a/final_architecture_decision.md:22-30, 211 |
| C1 CP-K, collision-probability operator (WL80A/B) | "PHYSICAL BASELINE ONLY", not a method | Learning-free and ALREADY KNOWN (= WL77A H3 at θ=0). Inadmissible as learned K under v0.1. Bias and cost unmeasured. "ARCHITECTURALLY INADEQUATE" is the arguable alternative | 80B_WORKLOG_C1_ARCHITECTURE_REASSESSMENT.md:114-123; 80A_WORKLOG_MULTI_AGENT_ARCHITECTURE_THOUGHT_EXPERIMENT.md:91, 103 |
| C2 SBG (WL80A) | Conditional learned secondary, not adopted | Moves the problem to coverage of an intermediate domain | 80A_WORKLOG_MULTI_AGENT_ARCHITECTURE_THOUGHT_EXPERIMENT.md:86, 104 |
| C3 CCA / C4 NSM (WL80A) | Rejected | C3 is enumeration that degenerates to per-port tabulation. C4 is a stored table of C1 columns, and its signed modal tail violates non-negativity | wl80a/final_architecture_decision.md:106-113 |
| DDT, Directional Delta Transport (WL84A Track A) | Withdrawn by its own advocate | The 4D outgoing state scales as α⁻⁴ (about 38 GB at α 0.05, DERIV). The break-even dirty fraction is below its geometric bound. The exact identity has no cost content | gm1_architecture_review/60_coordinator_synthesis.md:42, 46; 84A_WORKLOG_GM1_ARCHITECTURE_REVIEW.md:24 |
| Path-space / reservoir reuse (WL84A Track B) | Control only | A retracer with sampling memory; known prior art (GRIS/ReSTIR PT, residual path integrals) | 60_coordinator_synthesis.md:43; 84A_WORKLOG_GM1_ARCHITECTURE_REVIEW.md:25 |
| LPTG, lobe-partitioned transport graph (WL84A Track C) | Withdrawn | The sharp part is retraced per frame; the field aliases sharp-sender→rough-receiver transport; the dense re-solve makes S≈F; nothing global is learned | 60_coordinator_synthesis.md:44; 84A_WORKLOG_GM1_ARCHITECTURE_REVIEW.md:26 |
| WL84A overall | No mechanism novelty | Every central mechanism ALREADY KNOWN. No design holds sharp content in persistent state. Recommendation C+D | 60_coordinator_synthesis.md:50-72 |
| H-P, lobe-class partition + learned broad tail (WL86B) | Not selected; WEAKENED 2 / REFUTED 1 | Broad→narrow→broad links without visibility; moderate lobes have no owner; class boundary drawn post hoc at the failing panel; sharp part fully retraced | 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:93-110, 169-178 |
| H-0, explicit patch×bin on the same state, learning only in ER (WL86B) | The only unrefuted null | No learned global object, classical two-pass, no novelty. Candidate vs baseline is an owner decision | 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:112-117, 220 |
| H-E, keep the ECRO form; the recipe is the cause (WL86B) | Conditional | Frozen verdict NOT SUPPORTED. Resuming meets the no-rescue rule (owner decision). It fails if the global term still compensates under per-term targets, or if the non-negative rank is ≫16 | 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:119-124 |
| WL86B dropped variants: enclosure-bounded K, learned prolongation, control-variate table, defect correction, learned nonlocal-K generator | Rejected | Respectively: existing PRT; explicit refinement is 1.00-1.04×; about 1.2× gain; redundant given an exact operator; outside GM1 scope with negative transfer | 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:126-134 |
| Refuted-form summary (WL86B §A) | — | Dense learned coupling (cost); a learned K fit to non-passive, path-history-dependent targets and composed recursively; additive learned decompositions supervised on the sum only (four-part K, ECRO single+global) | 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:18-21 |
| Explicitly non-interchangeable with the contract | — | Free-space radiosity solver, SH2-only Lambertian model, scene-baked monolithic radiance MLP, SPT-1 repair | Architecture.md:103 |

## (D) Constraints any future architecture must satisfy

**Ownership and physics**
1. [FACT] Exclusive path ownership. [MEAS] Double claiming diverges: WL53 ρ 4.75 vs disjoint 0.42-0.82 (Architecture.md:25; dossier.md:426).
2. [FACT] Any recursively applied operator must be passive and correct on the fields the composition actually produces, including unsupervised ports and directions (Architecture.md:38; dossier.md:427).
3. [INF] It is not established whether this applies to a one-shot learned operator such as ECRO, which has no passivity constraint (dossier.md:428).
4. [FACT] No runtime inputs from target K, references, source identity, pose-specific oracles, baked radiance or T1 supervision. The rule stays source-independent and linear in the incident field (Architecture.md:58; RESEARCH_ROADMAP.md:16).
5. [FACT] A path-history-independent, validated reference is required (Architecture.md:23; BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:44, 48).
6. [FACT] Frozen ER is kept as the local layer. Changing ER needs a separate hypothesis (Architecture.md:44).
7. [FACT] Direction-selective transport needs directional incident information. Near-contact information must survive both K and C (Architecture.md:21, 36).

**Baseline gate**
8. [FACT] The mandatory comparison gate and the four-role separation apply. Beating a path tracer does not establish superiority over Neural Full Recompute (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:23-32).
9. [MEAS] Quality-matched path-traced re-render has been the binding economic bar so far:
   - WL67: 4 spp, 4.7 s vs 190 s;
   - WL69: 2.4 s vs 24 s;
   - WL70: 2.09 vs 2.40 s, fragile.
   (RESEARCH_ROADMAP.md:27-31)
10. [FACT] A GM1 run must include stale/no-update and same-mechanism full-rebuild controls (RESEARCH_ROADMAP.md:16).
11. [FACT] For a learned global candidate, an explicit same-state comparator is now expected: "matched explicit comparator" (RESEARCH_ROADMAP.md:128). [MEAS] WL87 showed explicit ≥ ECRO (RESEARCH_ROADMAP.md:135).

**Ray-tracing and lifecycle cost must not be hidden**
12. [FACT] The full lifecycle counts toward cost: selection, dependency maintenance, reconstruction, normalisation, cache prep, K/C application, solve and readout (Architecture.md:72). Separately: geometry prep, loading/checkpoint traffic, warm/cold time, unique memory and contention, and offline vs runtime (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:59).
13. [FACT] Learned coupling must not be "an exhaustive explicit per-surface full path tracer hidden under a neural name". Measure all visibility, ray and query costs (FNTO_ARCHITECTURE_CONTRACT_V0_1.md:86, 88). Falsification question: does it work "without an implicit dense ray/path tracing rebuild" (FNTO_ARCHITECTURE_CONTRACT_V0_1.md:150)?
14. [FACT] "Proving only an image match or implementing a full retracer does not complete GM1" (RESEARCH_ROADMAP.md:16).
15. [FACT] A state update that is cheaper only because its response is unacceptable does not satisfy the objective (Architecture.md:72).
16. [MEAS] Attribution caveat: WL86's 9× is two-tier estimator structure that a physically filled table also gets (86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:25). In ECRO v0, selective vs full state rebuild is only 20.8 vs 28.1 ms (dossier.md:438).

**Multi-model / multi-component justification**
17. [FACT] No canonical document literally says "no multi-model without justification". The closest canonical statements:
    - novelty cannot rest on "per-region neural models" (RESEARCH_CENTRIC_TOPIC.md:48);
    - multiple K terms must be "disjoint or explicitly weighted path families … document the path partition and any overlap correction" (FNTO_ARCHITECTURE_CONTRACT_V0_1.md:74);
    - shared mutable parameters are an explicit coupling to report (Architecture.md:76; AGENTS.md:109);
    - the F/S or Surface-Owned update semantics must not be combined silently (Architecture.md:78).
18. [MEAS]/[INF] The empirical basis is WL86B §C. In both the four-part K and ECRO, additive learned terms supervised only on the sum let a geometry-poor term absorb the deficit of a geometry-specific term (86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:52-80). WL86B criteria require a separately tallied target for every learned term, a class rule fixed beforehand from measured quantities, and a learned term that beats an explicit sweep of the same state (86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:222-228).

**Process**
19. [FACT] Choose one mechanism before its experiment. Freeze gates before results. No rescue, threshold change or automatic successor (Architecture.md:107; RESEARCH_ROADMAP.md:11, 128; BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:74; RESEARCH_CENTRIC_TOPIC.md:54).
20. [FACT] Each major decision must state its implications for rigid motion, local deformation, bending/twisting and self-contact (RESEARCH_CENTRIC_TOPIC.md:20).
21. [FACT] Batch classification: (1) a new architecture hypothesis is a regular numbered batch; (2) a dependent follow-up is a suffix; (3) a same-run correction is an amendment (AGENTS.md:3-11).
22. [FACT] Novelty needs correct ownership, validity under edits and measured quality/cost against prior art. A hierarchy, selective update, operator compression or BSSRDF approximation alone is not enough (RESEARCH_CENTRIC_TOPIC.md:48; FNTO_ARCHITECTURE_CONTRACT_V0_1.md:165).

**Document-consistency issues found (not resolved here)**
- [FACT] AGENTS.md:34-38, 52 and 114 still present FNTO v0.1, K_k44P and Surface-Owned units as current, with no mention of ER or ECRO. Architecture.md:9, 11, 13 and 91 supersede this.
- [FACT] Architecture §8 has no ECRO or C3UR disposition row.
- [FACT] BASELINE_ROLES has no REF-ER or REF-P row; it is dated 2026-11-02 (BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:3; dossier.md:449).
- [FACT] Open owner decisions recorded in the GM1 review and WL86B:
  - is a learning-free global solve around ER a candidate or a baseline?
  - does a learned sampling density count as "learned transport response"?
  - are multi-view/multi-frame workloads inside GM1?
  (60_coordinator_synthesis.md:94-100; 86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md:205-209)

Relevant paths: c:\Projects\ReST-GI\docs\Architecture.md, c:\Projects\ReST-GI\docs\RESEARCH_CENTRIC_TOPIC.md, c:\Projects\ReST-GI\docs\BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md, c:\Projects\ReST-GI\docs\RESEARCH_ROADMAP.md, c:\Projects\ReST-GI\docs\FNTO_ARCHITECTURE_CONTRACT_V0_1.md, c:\Projects\ReST-GI\docs\worklogs\86B_WORKLOG_INDEPENDENT_GLOBAL_TRANSPORT_ARCHITECTURE_STUDY.md, c:\Projects\ReST-GI\docs\plans\gm1_architecture_review\60_coordinator_synthesis.md, c:\Projects\ReST-GI\docs\plans\global_transport_study_86b\dossier.md, c:\Projects\ReST-GI\docs\worklogs\80B_WORKLOG_C1_ARCHITECTURE_REASSESSMENT.md, c:\Projects\ReST-GI\docs\plans\wl80a\final_architecture_decision.md