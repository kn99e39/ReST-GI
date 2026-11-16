# WL97 audit, track MAP: evidence dependency map and provenance (WL86-WL96A)

Independent reviewer report. Repository `c:/Projects/ReST-GI`, branch `Selective_Recompute`, HEAD `efcf86a` (clean tree at start). Server `LabServer63`. Audit date 2026-11-15 UTC (2026-11-16 KST). Nothing in the repository or in any `~/rest_*` run directory was modified; all outputs of this track are under `docs/audits/wl97_evidence_reliability/`, `experiments/audit97/MAP/`, `results/audit97/MAP/` and (server) `~/rest_x97/MAP/`.

Every statement below is labelled **[M]** measured / recomputed by this audit, **[R]** read from a stored artefact or source file (path given) without re-execution, or **[I]** inference.

## 1. Scope and method

1. **Executable revision of each evidence run [M].** On the server, `git hash-object` of every file in each run directory's `code/` tree (23 run dirs, 86-1,283 files each); locally, comparison with `git ls-tree -r <stated commit>`. A git blob id is content-addressed, so equality proves byte identity with the LF content of the committed file. For differing files the blob was searched in the whole history (`git log --all --find-object`). For the two run dirs with wholesale differences the comparison was repeated after `tr -d '\r'`.
2. **Protocol identity [M].** `git -c core.autocrlf=false show <commit>:<protocol> | sha256sum` against the `protocol=` field of `chain.status` (recorded from WL94 on).
3. **Artefact identity [M].** `sha256sum` on the server for the checkpoints, field / image references, masks and tables the P0 claims consume; comparison with every hash recorded in a worklog or result JSON, and of local `results/` copies with the server files.
4. **Dependency extraction [M + R].** A static parser (`extract_deps.py`) over the experiment code: cross-directory module imports, upstream run-dir artefact paths, uses of seed 86001 / state construction, uses of the `an94` metric helper. The chain scripts `run*.sh` were read for the symlink / copy set-up. Curated claim chains were entered from the worklogs and JSON cited per entry.
5. **Pre-freeze exposure [M].** JSON diff of the dev / probe protocol and code against the frozen versions for WL95A and WL96A; numeric comparison of the WL96A pre-freeze full-scale probe tables with the frozen-run tables.
6. **WL89 episode [R + M].** `chain.status`, artefact mtimes, `git diff 7515a29 5d6bb5c`, `amend1/audit89.json`, and the consumers found by step 4.

No GPU job was run. No training, no rendering. One CPU-only numpy comparison ran on the server (read-only inputs).

## 2. Dependency map

### 2.1 Current conclusions and where they come from

Sources of "current conclusions": `docs/RESEARCH_ROADMAP.md` l.128-156 (GM1 evidence record), `docs/Architecture.md` l.13 (ECRO lineage reinstated as development base, 2026-11-13; no candidate adopted), `AGENTS.md`, worklog verdict sections.

| conclusion / design constraint (roadmap wording, shortened) | worklog | measured quantity | protocol (sha256 LF) | executable | key inputs | reference | decision |
|---|---|---|---|---|---|---|---|
| Material rank-16 loses the sharp panel lobe; interface information sufficient | WL92 | region-C E_H 0.045 / 0.052; Eckart-Young 0.52 / 0.46 / 0.46 | `ecro_attr92` b558babd | `diag92.py`, `analyze92.py` @ 5e672ff | WL86 checkpoints, data, fields; M64 | WL86 REF / fields | gate STOP_INTERVENTION_G_rep |
| OWN supervision recovers ownership; full-system success not established | WL92A | val single-vs-L2 0.104 / 0.105; retention 0.21 | `ecro_own92a` 14fae630 | `own92a.py` @ c40226d | WL86 data, WL92 E4 files | WL92 fields | D_STOPPED (E4 not admitted, SNR 3.9 vs 4) |
| OWN residual = source-side once-scattered stage x tied factors | WL92B | counterfactual fractions f | `ecro_cf92b` db6c39fc | `cf92b.py`, `analyze92b.py` @ b29d8ff | OWN / SUM checkpoints, M64 substitution | WL92 / 92A fields | outcome C **by an implemented rule that differs from the written rule** (F-MAP-09) |
| Source-stage target not identifiable at preflight | WL93 | panel rel L1 0.508 vs bound 0.485 | `ecro_stage93` 68dcf525 | `prep93.py` @ c30c797 | OWN, WL92B E5 | q via M64 / Mq 1,024 | STOPPED |
| Smoke-scale source / dual-stage supervision: no signal / sharing signal | WL93A, B, C | pooled val debiased L2, WHOLE, SRC | 278af654 / 471179e3 / 9b725673 | `train93a/b/c.py` | WL93A data, 128 geometries, 2,000 steps, 2 seeds | q reference B5 | NO_SIGNAL / SHARING_SIGNAL / NO_SIGNAL |
| Learned S1 operator ~80 % wrong vs M64, geometry-common | WL93D | squared-error shares, C2 arm renders | `ecro_local93d` bb3fd47b | `ops93d.py` @ 4f3fc56 | OWN, WL93A T0, M64 | WL92B fields | labels REPRESENTATION_OR_OPTIMIZATION_DOMINANT + COADAPTATION_LIMITED |
| CK2 scene and its reference images | WL93D addendum | qualitative renders | addendum protocol | `checker93d.py` @ d8d092f | OWN | REF_ER_CK_* 4 x 8,192 spp | none (post hoc, qualitative) |
| H-M / H-A inconclusive; sharp deficit = panel-delivered input | WL94 | CK2 table / image metrics | `ecro_mat94` 99e199f3 | `mat94.py`, `an94.py`, `c94.py` @ e347904 | OWN, CK2, CKF_*, masks94 | CKF_*, REF_ER_CK_* | INCONCLUSIVE_SMOKE_FIT |
| Retention / exposure / query supervision smokes | WL94A, B, C | F2, MR, exposure 1.23x, sharp ROI | d0fd618c / 6b5b10c7 / 006941ac | `r94.py`, `b94.py`, `q94.py` | WL94 set | as WL94 | INCONCLUSIVE / STOP_INVALID_INTERVENTION / INCONCLUSIVE |
| Sender oracle supervision: no reliable improvement | WL94D | S1 field rel L1 0.82 -> 0.72 (W94-weighted) | `ecro_s94d` e0a38946 | `s94.py` @ a08c9bf | WL94C D models, M64 teacher | M64, CKF_* | NO_RELIABLE_IMPROVEMENT |
| Architecture selection premature; learned S1 operator at zero-operator level | WL95 (design) | op flux-L1 0.96-1.04 | none (design) | `results/wl95_calc/wl95_calc*.py` | `~/rest_x92/results/nmf.npz`, select94.json | M64 | PREMATURE |
| Rank 16 inadequate at grazing; binned sender biased 1.54x; exact per-ray closes it; learned S1 non-passive | WL95A | delivery ratio / rel L1, albedo_max | `ecro_attr95a` fb136ad2 | `a95.py` @ c152120 | OWN, CKF_none, REF_ER_CK_none, masks94, S1_ops.npz | CKF_none e2_p; REF_ER | three labels |
| No viable learned mechanism (scoped, provisional, dissent) | WL96 (design) | CPU oracles, Amdahl 1.09x / 1.16x | none | `results/wl96_calc/` scratch | WL95A numbers, WL91 life91.json | - | design verdict |
| No accuracy or lifecycle headroom for a learned >= 3-event resolvent on CK2 | WL96A | ROI paired change 0.0007-0.0008; U_ZERO 1.04x | `ecro_util96a` 0a593650 | `w96a.py` @ a1718f7 | CKF_none, REF, masks94, FZ_own, T95A | CKF_none e3; REF_ER | S1_NO_HEADROOM |

### 2.2 Batch-level edges (consumer <- upstream)

Full machine-readable version: `results/audit97/MAP/dependency_map.json` (`batch_edges`, `static_code_dependencies`).

| consumer | upstream numbers / tensors consumed |
|---|---|
| WL86A | WL86 checkpoints |
| WL87 | WL86 fields, renders, `final.json`, `qualify.json`; WL86A `D_SPLIT_*`, importance |
| WL88 | WL87 `X_E2_T1.npz` |
| WL89 | WL88 code incl. the `tests88.T_EDIT` gate |
| WL90 | WL89 `w_choice.json`, `F89_*`, `ORD_*`, masks (all produced before the amendment), `stage3_89.json` (after) |
| WL91 | WL89 `w_choice.json`, `F89_*`, `ORD_*`, masks, `REF_ER_*`; WL90 `F_W1_T1.npz` |
| WL92 | WL86 data, fields, masks, models, renders, `final.json`; amended `pt86a.field_src`; M64 from `quad87` |
| WL92A | WL86 data; WL92 E4 files |
| WL92B | WL92A checkpoints, tables, `final92a.json`; WL92 E4 files |
| WL93 | WL92A own checkpoints, `exclusion.json`; WL92B `final92b.json`, E5, CF tables |
| WL93A / B / C | WL93 `val_q`; WL93A models / data / `taskA.json`; WL92B fields, `CF_E2_T1.npz` |
| WL93D | WL92A own; WL93A T0; WL92B `repro.json`, CF tables |
| WL93D addendum | WL93D tables; produces CK2 scene and `REF_ER_CK_{none,bx38,bx75}` |
| WL94 | addendum scene + 12 reused reference renders + CK tables; WL93D `I_E2_T1.npz`; WL92A own; WL92B `repro.json` |
| WL94A-D | WL94 data, fields, masks94, FZ_own, select94, stageB; each other's models / audits (94D <- 94C <- 94B <- 94A) |
| WL95 (design) | WL92 `nmf.npz` (M64, NMF16); WL94 `select94.json`; WL94D numbers |
| WL95A | WL94 `CKF_none`, masks94, `REF_ER_CK_none` (addendum origin), `FZ_own`; WL94D `audit.json`, `SN_own_*`; WL95 `S1_ops.npz`, `wl95_calc2.json`; WL92A own |
| WL96 (design) | WL95A numbers; WL91 `life91.json`; WL90 P2plus; WL87 counts |
| WL96A | WL94 `CKF_none`, masks94, `REF_ER_CK_none`, `FZ_own`; WL95A `T95A.npz`, floor render; WL92A own (timing) |

**Numbers quoted from earlier worklogs rather than recomputed, and my check [M]:**

| quoted in | number | stored source | result |
|---|---|---|---|
| WL96A protocol `pre_freeze_exposure` | WL95A G_s8601 / G_s8602 E_H 0.0251 / 0.0150, floor 0.0118 | `results/ecro_attr95a/an.json` images | 0.0251 / 0.0150 / 0.0118 - matches |
| WL96 worklog l.145-146 | 124.7 / 75.8 ms, 1.09x / 1.16x | `results/gm1_c3urx1_91/results/econ/life91.json` C3URX1 edit 0.12466 s, source 0.07581 s | matches; cross-workload (GM1 V3, not CK2), disclosed |
| WL94D worklog l.109 | WL92 rank-16 Eckart-Young 0.46-0.52 | `results/ecro_attr92/results/local.json` S1_panel 0.520 / 0.458 / 0.461 | matches (WL95 later states the comparison used unmatched norms) |
| WL95A T_REPRO94D | WL94D CK2 flight floor 1.378 / 1.211 / 1.206 | recomputed inside WL95A stage 0, `s0.json` max_abs_dev 1.48e-7 | recomputed by the batch, not quoted |
| WL93 | WL92B outcome | re-derived into `results/ecro_stage93/recon92b.json` | official C vs written rule D / INCONCLUSIVE (F-MAP-09) |

### 2.3 Shared dependencies (single points of failure)

| id | dependency | consumers | what breaks if it is wrong | status in this audit |
|---|---|---|---|---|
| SD1 | learned ER closure (WL83/84) | all | all references and M64 live in an ER world | identity only |
| SD2 | `scene86` / `pt86` tracer, 4 x 16 bins, 1,270 patches | all | everything | never modified after 7786d09 [M] |
| SD3 | state realization seed 86001 - **one** ray-offset pattern and **one** 16-point source pattern broadcast to every patch (`pt86.py:114-127`) | WL86 training data; every "deployed" table; M64 | every single-draw "deployed" number is one draw of a patch-correlated estimator | code read [R]; WL95A measured deployed 1.21x vs pooled 1.54x [R] |
| SD4 | M64 = `quad87.Quad(realization_us(86001), 65536 rays seed 87001, 16384 dense seed 87002).enc_matrix()` (`diag92.py:133-139`) | WL87, WL92, WL92B, WL93-WL94D (teacher / "physical" substitution), WL95, WL95A, WL96A | every result that treated M64 substitution as ground truth | construction traced [R]; `quad87` / `xport87` unmodified since 60efecb [M] |
| SD5 | `pt86a.field_src` event split (amended 1bd7df1; panel tag = material id 1 at the k = 2 vertex) + `mat94` run-time regex patch | WL92-WL96A field references | all component attributions | diff read [R]; WL94 identity tests quoted [R] |
| SD6 | OWN checkpoints own_s8601 / own_s8602 | WL92B, 93, 93D, addendum, 94-94D, 95A, 96A | P0-1 and every "learned" row of P0-2..4 | sha256 recomputed [M] |
| SD7 | CK2 scene (one grazing point light), sha 977ba122 | WL94-WL96A | P0-2, P0-3, P0-4 and the WL95A part of P0-1 | hash chain verified [M] |
| SD8 | `REF_ER_CK_none_CK2_T1_r0..3` (4 x 8,192 spp, seeds 86100-86103), produced by the WL93D addendum | WL94-WL96A | all ROI E_H values and noise floors | sha256 identical in both run dirs and `reused_refs.txt` [M] |
| SD9 | `CKF_none.npz` (2 x 4,096 spp, seeds 94301 / 94302) | WL94, 94D, 95A, 96A | P0-2, P0-3, P0-4 | sha256 = WL96A record [M] |
| SD10 | `masks94.npz` | WL94-WL96A | all CK2 region metrics | identical in three run dirs [M] |
| SD11 | `an94.W` (coded 8 x 8; true 4 x 16) and `an94.eh` | W: WL94, 94A, 94B, 94C, 94D; eh: WL94-96A | S1-field (non-Lambert) metrics of WL94-94D | defect confirmed [M], see F-MAP-08 |
| SD12 | WL86 dataset (1,024 train / 64 val; server only) | every trained model | all learned results share one distribution | counts only |
| SD13 | WL89 workload V3 and WL91 timing | WL90, WL91, WL96 design estimate | WL96 cost prediction only | anchors verified [M] |
| SD14 | server-only storage of all P0 inputs | all | silent change undetectable except through recorded hash prefixes | all prefixes match [M] |

## 3. Findings

**F-MAP-01 - confirmation - material: yes.** *Every evidence run executed the stated revision.* 23 run directories, 11,934 file comparisons (sum of `n_server_files` in `code_revision_check.json`; an earlier draft of this report said "13,6xx", which was a miscount): every deployed file is byte-identical to the stated commit, except (a) the documented post-freeze / post hoc files of F-MAP-03 and (b) WL86 / WL86A, whose archives were deployed with CRLF line endings and are identical after CR removal (85 / 85 files vs 63972c8; 98 / 98 vs bd81442, 97 / 98 vs f6483ab - the remaining file is the post hoc `viz86a2.py`). The shared production modules (`ecro86/*.py`, `gm1_explicit87/*.py`, `ecro_mat94/{mat94,an94,c94}.py`, `ecro_attr92/*.py`) were never modified after their freeze commits, so every later batch imported the same bytes that produced the earlier evidence; the only shared module changed after first use is `pt86a.py` (amendment 1bd7df1, adds `split=True`). Evidence: `results/audit97/MAP/code_revision_check.json`, `crlf_check.json`. Proposed verdict: VERIFIED. Confidence: high.

**F-MAP-02 - confirmation - material: yes.** *Protocol hashes.* For WL94, 94A, 94B, 94C, 94D, 95A, 96A the `protocol=` prefix in `chain.status` equals sha256 of the LF content at the freeze commit (99e199f3, d0fd618c, 6b5b10c7, 006941ac, e0a38946, fb136ad2, 0a593650). No protocol file changed between its freeze commit and HEAD except WL89 (amendment, expected). VERIFIED. High.

**F-MAP-03 - provenance_gap - material: no.** *Files changed or added in run dirs after the freeze* (blob -> commit mapping):
- WL88 `analyze88.py`, `viz88.py` = commit c5e5e2f (logged `POSTFIX_START` in `chain.status`, after `CHAIN_FAILED`).
- WL93 `analyze93.py` = 56be753, edited 06:45:08Z, 67 s after `STOP_TARGET_NOT_IDENTIFIABLE` was logged; the diff is one None-handling expression on the stop path (worklog l.181-187, "Amendment 1 awaits owner approval"). The gate value comes from `ident.json` (06:44:00Z, frozen code).
- post hoc scripts: WL86A `viz86a2.py`, WL93 `posthoc93.py` / `vizgate93.py`, WL93A `posthoc93a.py` / `vizpost93a.py`, WL93D `vizrender93d.py`, WL94B `posthoc94b.py` / `figb4_posthoc.py`, WL96A `figw3_posthoc.py` (run dir root).
- WL86A's own code directory was overwritten by the two amendment deployments (mtime 05:53:26Z), so the pre-amendment `pt86a.py` that ran Stage A-C is no longer on the server; the repository copy at 46593f0 cannot be byte-checked against what ran.
- chain.status of WL86-WL93D records no protocol hash; uncommitted debug scripts sit in `~/rest_rel89/` (`dbg89.py`, `dbg89b.py`, `dbgp.py`, `dbgs.py`).
None of these changes the computation that produced a gate value. Proposed verdict: NONE (hygiene). High.

**F-MAP-04 - confirmation - material: yes.** *Artefact identities of the P0 inputs* (sha256 prefix, this audit): `own_s8601.pt` 739e7eb5, `own_s8602.pt` cee9a09a (= WL92B `repro.json`, = WL94 `pre.json` T_CKPT, = WL94 worklog); `CKF_none.npz` 3a6238a8, `masks94.npz` b9a7d46b (identical copies in rest_x94 / x95a / x96a), `FZ_own_s8601/2.npz` 44fade05 / 03183dd2, `T95A.npz` d009c57b, `T96A.npz` 8019cb39 (all = WL96A worklog l.222-225); `REF_ER_CK_none_r0..3` 89839c42 / 85e8176b / a701f32b / 84fe5927 (rest_x93d_ck2 = rest_x94 = `reused_refs.txt`); `S1_ops.npz` 8654ae79 (server = repo = WL95A worklog); `scene_ck2.json` 977ba122 (repo = addendum chain.status = WL94 `pre.json`). Local `results/ecro_attr95a/*` and `results/ecro_util96a/*` are bit-identical to the server files. VERIFIED. High.

**F-MAP-05 - dataset_limitation - material: yes - P0-2, P0-3, P0-4 (and the WL95A part of P0-1).** *All four rest on one scene, one light, one reference set, two checkpoints.* The CK2 scene was introduced by the WL93D checkerboard addendum (labelled post hoc, qualitative, non-numbered; five commits in 45 minutes, d8d092f = revision 2). Its four reference renders are reused, by file copy, through WL94, WL94A-D, WL95A and WL96A; the field reference `CKF_none` (two 4,096-spp halves) is likewise single. WL95A did not use its planned secondary workload ("no semantically identical panel-tagged split was verified", worklog l.15). Mitigations found: WL94 `pre.json` T_REF_REPRO re-rendered one addendum reference with the WL94 renderer to 7.7e-7, T_ID_TRACE 8.9e-6, T_ID_FIELD 3.7e-8 [R]. Decision-level impact: the P0-2..4 magnitudes (1.54x, 0.111, 0.0007, 1.04x) are measurements of one configuration; they do not carry an across-scene uncertainty. Proposed verdict: CONDITIONALLY_VALID. High for the dependency structure.

**F-MAP-06 - dataset_limitation - material: yes - P0-1, P0-2, earlier attributions.** *The "deployed" state is one sample pattern shared by all patches.* `pt86.py:114-127` draws `us` with shape (4, 1, K) and `u_src_np` with shape (2, 1, PSRC) from `default_rng(86001)` and broadcasts both to all N patches. All 42 panel patches therefore use the same 16 relative source points and the same 8 directions per bin; the estimator error is correlated across patches and does not average out in a region metric. WL95A quantified it on CK2: deployed M64 box-A delivery ratio 1.208 against a 64-draw expectation of 1.544 (SE of rel L1 0.035), and a single 16-point source draw is 0.117 (max 0.28) rel L1 from the pooled source. Every table-level comparison of WL86-WL94D that used the deployed state inherits this (WL92 "B SUPPORTED 0.045 / 0.052", WL92B counterfactual fractions, WL93C oracle, WL93D C2 arms, WL94 C2_srcS1, WL94D teacher and flight floor). WL92 itself reports "about half of that is the deployed state-draw sampling limit". Decision-level impact: pre-WL95A statements of the form "arm X is / is not better than exact M64" are single-draw statements. Proposed verdict: CONDITIONALLY_VALID. Medium-high (code read and WL95A numbers; not re-measured).

**F-MAP-07 - invalid_inference (risk) - material: yes - P0-1, P0-5.** *M64 served as "physical truth" before it was shown to be biased.* WL92B ("exact q closes 96 % of the seed gap"), WL93C (ORACLE_FEASIBLE), WL93D ("~80 % wrong vs frozen-ER M64") and WL94D (teacher) measure distance to M64 or to M64-propagated quantities. WL95A then shows the binned M64 sender over-delivers 1.54x on the grazing case (physical q256: 1.42x) and that at the delivery level the learned s8601 sender (abs rel L1 0.569) and the M64 sender (0.546) are equally far from the reference, with opposite sign (ratio 0.52 vs 1.54) [R, `an.json`, worklog B2]. The sender-field error of the learned operator against the physical q256 remains large (1.08 / 1.19 rel L1, energy 0.42 / 0.86) [R], so P0-1's first half does not depend on M64 alone; but "avoidable" is shown only by oracle constructions (rank 32: 0.54, M64: 0.24 vs q256), not by any trained model, and the delivery-level benefit of removing it inside a 64-bin carrier is not demonstrated. Proposed verdict: CONDITIONALLY_VALID for P0-1; UNTESTED_INFERENCE for any reading "a correctly trained S1 sender would fix delivery". Medium.

**F-MAP-08 - implementation_defect - material: yes - WL94D label, P0-1 (WL94D part).** *`an94.W` uses the wrong bin layout* (independently confirmed): `an94.py:19` `NB, NPHI, NZ = 64, 8, 8`, `:32` `W = ((d // 8) + 0.5) / 8 * dOmega`; the representation is 4 rings x 16 sectors (`ecro86.json` `bins_z: 4`, `bins_phi: 16`; `scene86.py:15`). Ratio to the correct weight: 0.5 / 1.5 for sectors 0-7 / 8-15 of the grazing ring, 0.83 / 1.17, 0.9 / 1.1, 0.93 / 1.07 for the next rings; the sum (pi) and any azimuth-uniform field are unaffected. Consumers [M, static scan]: `an94.ex` (WL94 / 94A table metrics), `b94.py:47` (WL94B stored exitance; the WL94B exposure gate uses the unweighted loss-mass terms `nm` / `nt`, `b94.py:81-83`, so it is not affected [R]), `s94.py:129` (G_PHYS), `:157` (CK2 flight floor), `:287-312` (S1 field metrics that feed the frozen `SENDER_IMP` mapping). The WL94D teacher loss itself is unweighted (`s94.py:57-60`). WL95 documented the defect; WL95A and WL96A use a corrected `W_T` and re-measured the OWN sender; **no later batch re-derived the WL94D frozen mapping (T / R / P candidates) under the corrected weight.** Decision-level impact: WL94D's NO_RELIABLE_IMPROVEMENT rests on a mis-weighted non-Lambert field metric; whether the label changes is unknown. Proposed verdict: IMPLEMENTATION_DEFECT (confined to WL94-series S1-field metrics). High for the defect, low for the direction of its effect.

**F-MAP-09 - evaluation_defect - material: yes - P0-1 lineage, P0-5.** *WL92B's official outcome was produced by an implemented rule that differs from its frozen written rule.* `results/ecro_stage93/recon92b.json` [R]: written rule -> H_interact INCONCLUSIVE, outcome "D (code fall-through; the written mapping has no clause ...)"; implemented rule (only X_M12 and X_L3 checked for dominance, although X_L2 and X_Q are dominant in both E2 cases) -> H_interact SUPPORTED, outcome C. The roadmap entry for WL92B still reads "Outcome C ... by the frozen mapping" and "H_interact SUPPORTED"; the reconciliation is mentioned only inside the WL93 entry. WL92B's attribution is the origin of the source-stage / S1-sender line (WL93 -> WL94D). Measurements are unaffected. Proposed verdict: CONDITIONALLY_VALID (measurements) / label not supported by the frozen written rule. High (stored record), not re-derived from `final92b.json` by this audit.

**F-MAP-10 - provenance - material: yes for WL89-WL91, no for P0.** *WL89 "invalid comparison" episode.* Timeline [R, `~/rest_rel89/chain.status`]: start 03:55:44Z at 7515a29; workload V3 chosen 04:07:26Z; 04:35:33Z `GATE TESTS FAIL` -> `STOP STOPPED_INVALID_COMPARISON` (only failing test: WL88's `tests88.T_EDIT` strict digest of the R1 relation); code redeployed at 5d6bb5c 06:01:21Z; `RESUME` 06:01:30Z with all finished stages skipped through their rc files; tests rerun -> PASS 06:02:59Z; done 06:36:46Z. The amendment replaces the predicate (strict digest over all stored entries -> operational digest over valid samples and operator tensors) and adds `audit89.py`; `rel89.py`, `run89.py`, `fx89.py`, thresholds, workload and references are untouched [M, `git diff --stat`]. Evidence for the false positive [R, `amend1/audit89.json`]: 2 stored `dbin` entries differ, both invalid in both relations; operational digests, operator tensors and CPU transport outputs bitwise equal; perturbing 157,538 invalid entries changes nothing; a first audit run had a test bug and was rerun (`audit89_run1_testbug.json` kept). Original stop preserved in `results/orig_stop/`. The human approval is asserted in the protocol text and cannot be verified from repository artefacts.
Later reliance: **WL90** - pre-amendment artefacts (`w_choice.json`, `F89_*`, `ORD_*`, masks) plus post-amendment `stage3_89.json` (`analyze90.py:169`); **WL91** - pre-amendment artefacts incl. `REF_ER_*`, plus WL90's `F_W1_T1.npz`; **WL96 (design)** - WL91 `life91.json` timing (verified); **WL92-WL95A, WL96A** - none. All WL89 comparative quantities (C3UR vs R0, floor 0.038 vs oracle 0.014, economics) exist only under the amended gate; the pre-amendment artefacts reused downstream were generated by code the amendment did not touch. Not re-executed by this audit. Proposed verdict: CONDITIONALLY_VALID. Medium.

**F-MAP-11 - statistical_weakness - material: yes (interpretation of "frozen") - P0-4.** *WL96A outcome tables existed before the freeze.* Dev chain `~/rest_x96adev` 16:55-16:58Z; full-scale probe `~/rest_x96aprobe` 16:58-17:04Z (tails, renders, Stage 2 fit, lifecycle cost); freeze commit 17:04:49Z; chain 17:05:02-17:09:01Z; results commit 17:13:06Z. [M] The probe's `T96A.npz` equals the frozen run's to rel L1 <= 3.4e-9, rel max 7.7e-8 (6 of 12 arrays bitwise equal); tail cost 3.42 vs 3.31 ms. [M] Between dev / probe and the frozen protocol only `execution` and `pre_freeze_exposure` differ; `w96a.py` is identical. So the criteria were not changed after exposure - this supports the worklog's disclosure (l.249-259) - but the decisive numbers were computable from files that existed before the freeze, and "outcome lines were suppressed" is not verifiable. Independently, the thresholds (minimum meaningful effect 0.005; lifecycle rule 1.25x) were written by the author who had WL96's predictions (headroom ~0.001; 1.09x / 1.16x) in hand, so the test could hardly have come out otherwise: it is a confirmation of a prediction, not a risky test. WL95A: same pattern at reduced scale (dev 21:00-21:01Z, freeze 21:01:38Z, criteria unchanged [M]). Decision-level impact: none on the numbers; "frozen before evidence" should be read as "criteria frozen before the code ran". Proposed verdict: CONDITIONALLY_VALID. High.

**F-MAP-12 - confirmation - material: no.** Cross-batch quoted numbers checked in section 2.2 all match their stored sources.

**F-MAP-13 - provenance_gap - material: yes for reproducibility.** OWN checkpoints, `CKF_none`, `masks94`, `FZ_own`, the reference renders, `T95A.npz` and the 2.6 GB WL86 dataset exist only on LabServer63 under one user account; the WL95 design inputs live in `~/wl95_calc/`, outside the `rest_*` convention (its `wl95_calc.py` is the committed file with CRLF endings [M]). There is no hash manifest other than the prefixes scattered in worklogs. Proposed verdict: NONE (risk note). High.

**F-MAP-14 - invalid_inference (scope) - material: yes - P0-5.** The negative chain WL92-WL94D is dominated by non-decisive labels on one lineage: D_STOPPED (92A), STOPPED (93), NO_SIGNAL_AT_SMOKE_SCALE (93A, 93C), INCONCLUSIVE (94, 94A, 94C), STOP_INVALID_INTERVENTION (94B; the intervention was never trained), NO_RELIABLE_IMPROVEMENT with "representation not tested" (94D). All trained arms are 1,000-2,000-step continuations or 128-geometry smokes from two seeds of one 735,664-parameter recipe on one dataset (SD6, SD12). The decisive-looking results are learning-free (WL95A, WL96A) and single-workload (SD7). They support: (i) a tied rank-16 a c^T local factor cannot carry the grazing S1 lobe on CK2 as fitted (WL96 itself reclassifies this as "fit allocation ... moot"); (ii) a 64-bin patch carrier has a systematic delivery bias there; (iii) behind a path-traced prefix the >= 3-event broad tail leaves no room on CK2. They do not test a retrained or differently parameterised learned operator at full scale. Proposed verdict: first clause of P0-5 CONDITIONALLY_VALID only for the specific fitted instances and carrier resolution; "rejecting the mechanisms" in general is UNTESTED_INFERENCE; second clause (no general impossibility) is consistent with the evidence. Medium (structural reading, no re-measurement).

## 4. P0 assessment within the MAP lens (provenance and dependency structure)

| claim | verdict | restrictions that the dependency structure imposes |
|---|---|---|
| P0-1 | CONDITIONALLY_VALID | code, inputs and numbers trace cleanly (F-MAP-01, 04). Two checkpoints of one recipe; magnitudes seed-dependent. "Avoidable" = oracle factorisations, not a trained model (F-MAP-07). Passivity = max over 42 patches x 64 incident bins x 3 channels of the S1 local-factor directional albedo at the CK2 geometry (`a95.py:147-152, 316-318`); mean albedo and the fraction of violating rows are not stored for the learned arm, and no system-level (spectral-radius / energy-balance) test of the deployed operator exists in this chain, so "may violate" is the correct strength. WL94D's contribution is weakened by F-MAP-08; WL92B's by F-MAP-09 |
| P0-2 | CONDITIONALLY_VALID | one scene, one grazing light, box-A exitance; 64 pooled draws; reference half-noise 0.059; SD7-SD10 shared with P0-3 / P0-4 |
| P0-3 | CONDITIONALLY_VALID | true for the 64-draw mean only (ratio 1.022, rel L1 0.111 +- 0.041); per draw the exact estimator is worse than the binned one (median 0.617, worst 2.21); image-level difference to the floor within noise; same single workload |
| P0-4 | CONDITIONALLY_VALID | CK2 only; ceiling argument with a path-traced prefix, no learned resolvent trained; lifecycle on one GPU with CPU-bound overhead; outcome tables pre-existed the freeze and thresholds were set with the prediction known (F-MAP-11) |
| P0-5 | CONDITIONALLY_VALID (first clause partly UNTESTED_INFERENCE; second clause consistent) | F-MAP-14 |

## 5. Special checks

| check | result | evidence |
|---|---|---|
| server code = stated commit, all 23 run dirs | PASS (CRLF-only deployments for WL86 / WL86A; listed post-freeze files only) | `code_revision_check.json`, `crlf_check.json` |
| protocol sha256 in chain.status (WL94-WL96A) = LF content at freeze | PASS, 7 / 7 | section 3 F-MAP-02 |
| shared modules unmodified after first use | PASS except `pt86a.py` (amendment 1bd7df1, before any WL92+ use) | `git log -- experiments/<dir>` |
| OWN / CKF_none / REF / masks94 / FZ_own / T95A / T96A identities | PASS, all recorded prefixes reproduce | F-MAP-04 |
| local results = server results (WL95A, WL96A) | PASS, bit-identical | sha256 lists |
| WL96A criteria unchanged between pre-freeze runs and freeze | PASS | JSON diff |
| WL96A probe vs frozen tables | equal to 7.7e-8 rel max | `cmp96_probe_vs_frozen.py` |
| `an94.W` layout | DEFECT confirmed | F-MAP-08 |
| WL89 amendment touched generator code | NO | `git diff --stat 7515a29 5d6bb5c` |
| WL96 timing anchors = WL91 stored values | PASS | `life91.json` |

## 6. Not verified, and why

- No measurement was re-executed (no GPU job): WL95A delivery numbers, the passivity statistic, WL96A tails and lifecycle, the WL89 amendment audit and the WL94 identity tests are read from stored artefacts. The track brief was provenance; other tracks own re-measurement.
- The effect of the `an94.W` defect on the WL94D frozen mapping (needs the T / R / P candidate sender fields under `W_T`; server GPU).
- WL92B's written-rule outcome was taken from `recon92b.json`, not re-derived from `final92b.json`.
- The correctness of M64 construction (`quad87.enc_matrix`), of the learned ER closure and of the event-split logic was not reviewed; only their identity and consumers.
- Human approvals (WL89 amendment 1; pending WL93 amendment 1) are not verifiable from repository artefacts.
- WL52 was not opened: no primary conclusion of WL92-WL96A consumes a WL52 artefact or number (static scan found no edge); WL96's "becomes learned C" argument cites it only by analogy.
- WL86-WL88 results were checked at the identity level only.

## 7. Reproducibility appendix

Local (Git Bash, repo root):

```
python experiments/audit97/MAP/compare_blobs.py     # needs results/audit97/MAP/server_blobs/blobs_*.tsv
python experiments/audit97/MAP/compare_crlf.py      # needs norm_rest_ecro86.tsv, norm_rest_ecro86a.tsv
python experiments/audit97/MAP/extract_deps.py      # static dependency scan -> static_deps.json
python experiments/audit97/MAP/build_map.py         # -> dependency_map.json
git -c core.autocrlf=false show <commit>:<protocol path> | sha256sum
```

Server (`ssh LabServer63`, everything written under `~/rest_x97/MAP/`):

```
# blob manifests (per run dir d):
cd ~/d/code && find . -type f ! -path '*__pycache__*' ! -name '*.pyc' ! -path './docs/*' ! -path './results/*' | LC_ALL=C sort > L
git hash-object --stdin-paths < L ; xargs -d '\n' stat -c '%y %s' < L        # -> blobs_<d>.tsv
tr -d '\r' < file | git hash-object --stdin                                   # -> norm_<d>.tsv (rest_ecro86, rest_ecro86a)
sha256sum ~/rest_x92a/results/models/own_s860?.pt ~/rest_x94/results/fields/CKF_none.npz ~/rest_x94/results/pre/masks94.npz \
          ~/rest_x94/results/tables/FZ_own_s860?.npz ~/rest_x94/results/renders/REF_ER_CK_none_CK2_T1_r?.npz \
          ~/rest_x95a/results/tables/T95A.npz ~/rest_x96a/results/tables/T96A.npz ~/wl95_calc/S1_ops.npz
diff <(python3 -m json.tool ~/rest_x96adev/code/experiments/ecro_util96a/protocol/ecro_util96a.json) \
     <(python3 -m json.tool ~/rest_x96a/code/experiments/ecro_util96a/protocol/ecro_util96a.json)
. ~/rest_x95a/env.sh; CUDA_VISIBLE_DEVICES= $PY ~/rest_x97/MAP/cmp96.py       # = experiments/audit97/MAP/cmp96_probe_vs_frozen.py
```

No seeds were used by this track (no stochastic computation). Output files: `results/audit97/MAP/{dependency_map.json, code_revision_check.json, crlf_check.json, static_deps.json, static_deps.txt, server_blobs/*.tsv}`.

## 8. Completion check after the interrupted first attempt (2026-11-16)

The first attempt of this track was interrupted at the moment of reporting. On resumption the report was read once and checked for completeness (sections 1-7 present, final section not truncated, `dependency_map.json` parses and holds 12 top-level blocks incl. 23 run records, 24 protocol hashes, 18 artefact identities, 25 batch edges, 14 shared dependencies, the WL89 episode and 5 P0 claim chains). No work was redone. Spot-checks against stored outputs and sources [M]:

| decisive number in this report | re-check | result |
|---|---|---|
| protocol sha256 (LF) at freeze: WL96A 0a593650, WL95A fb136ad2, WL94 99e199f3, WL94D e0a38946 | `git -c core.autocrlf=false show <commit>:<path> \| sha256sum`; `results/ecro_util96a/chain.status`, `results/ecro_attr95a/chain.status` | all four reproduce and equal the `protocol=` field |
| artefact sha256: own_s8601 739e7eb5, own_s8602 cee9a09a, CKF_none 3a6238a8, masks94 b9a7d46b, T95A d009c57b, T96A 8019cb39 | `sha256sum` on LabServer63 (read-only) | all six reproduce |
| WL89 timeline (FAIL / STOP 04:35:33Z, RESUME 06:01:30Z at 5d6bb5c, PASS 06:02:59Z, done 06:36:46Z) | `grep` on `~/rest_rel89/chain.status` | reproduces |
| `an94.W` layout defect (F-MAP-08) | `experiments/ecro_mat94/an94.py:19,32`; `ecro86.json:47` (`bins_z 4`, `bins_phi 16`); `scene86.py:132-137` (bin = iz * NPHI + iphi, z = (iz + 0.5) / NZ) | confirmed; weight ratios 0.5 / 1.5 (ring 0), 0.83 / 1.17, 0.9 / 1.1, 0.93 / 1.07 follow arithmetically |
| WL95A numbers: deployed 1.208, pooled 1.544, exact 1.022 / 0.111, E_H 0.0251 / 0.0150 / 0.0118 | `results/ecro_attr95a/an.json` (`delivery/dep\|A4\|boxA`, `pool\|A4\|boxA`, `pool\|X1\|boxA`, `images/*`) | 1.2082, 1.5441, 1.0224 / 0.1106, 0.02512 / 0.01504 / 0.01184 |
| WL91 timing anchors 124.7 / 75.8 ms | `results/gm1_c3urx1_91/results/econ/life91.json` | 0.124655 / 0.075812 s |
| WL92B rule discrepancy (F-MAP-09) | `results/ecro_stage93/recon92b.json` | as stated (official C / SUPPORTED vs written rule D / INCONCLUSIVE) |
| file-comparison count | sum of `n_server_files` | 11,934 - the only correction made to the report (F-MAP-01 said "13,6xx"); no finding or verdict changes |
