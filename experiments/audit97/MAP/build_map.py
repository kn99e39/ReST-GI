#!/usr/bin/env python3
"""WL97 MAP: assemble results/audit97/MAP/dependency_map.json.

Machine-derived parts are loaded from this track's own outputs (code_revision_check.json, crlf_check.json, static_deps.json);
curated parts (claim chains, shared dependencies, artifact identities, WL89 episode) were entered by the auditor from the
sources cited in each entry (file:line, server path, sha256 prefix). Nothing in the repository is modified.
"""
import json, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
O = os.path.join(REPO, "results", "audit97", "MAP")
L = lambda n: json.load(open(os.path.join(O, n), encoding="utf-8"))

PROTO = {  # worklog: [protocol path, freeze commit, sha256[:16] of LF content at freeze, hash recorded in chain.status or None]
    "WL86": ["experiments/ecro86/protocol/ecro86.json", "7786d09", "ecca67246ef013d7", None],
    "WL86-amend1": ["experiments/ecro86/protocol/amendment1.json", "63972c8", "032a624e61ef10fd", None],
    "WL86A": ["experiments/ecro86a/protocol/ecro86a.json", "46593f0", "9379c2fe2bc58c6d", None],
    "WL87": ["experiments/gm1_explicit87/protocol/gm1_explicit87.json", "60efecb", "b86f3abf446b932f", None],
    "WL88": ["experiments/gm1_relation88/protocol/gm1_relation88.json", "26e13f2", "0ccb32866be26f2e", None],
    "WL89-freeze": ["experiments/gm1_relation89/protocol/gm1_relation89.json", "7515a29", "f7bb6a69d0007eaf", None],
    "WL89-amend1": ["experiments/gm1_relation89/protocol/gm1_relation89.json", "5d6bb5c", "9a5b460462205d62", None],
    "WL90": ["experiments/gm1_gap90/protocol/gm1_gap90.json", "51b93b8", "19b4135a45a6ee63", None],
    "WL91": ["experiments/gm1_c3urx1_91/protocol/gm1_c3urx1_91.json", "7a8544e", "1cf84cfd521e1794", None],
    "WL92": ["experiments/ecro_attr92/protocol/ecro_attr92.json", "5e672ff", "b558babd40388abf", None],
    "WL92A": ["experiments/ecro_own92a/protocol/ecro_own92a.json", "7247d30 (= c40226d)", "14fae630a7a3b367", None],
    "WL92B": ["experiments/ecro_cf92b/protocol/ecro_cf92b.json", "b29d8ff", "db6c39fc18108c8d", None],
    "WL93": ["experiments/ecro_stage93/protocol/ecro_stage93.json", "c30c797", "68dcf525943082fe", None],
    "WL93A": ["experiments/ecro_stage93a/protocol/ecro_stage93a.json", "888ceb6", "278af654eb89171b", None],
    "WL93B": ["experiments/ecro_stage93b/protocol/ecro_stage93b.json", "03fbc84", "471179e3c4601b32", None],
    "WL93C": ["experiments/ecro_stage93c/protocol/ecro_stage93c.json", "cab6e18", "9b7256739890ebca", None],
    "WL93D": ["experiments/ecro_local93d/protocol/ecro_local93d.json", "4f3fc56", "bb3fd47b03080969", None],
    "WL94": ["experiments/ecro_mat94/protocol/ecro_mat94.json", "e347904", "99e199f3847939e8", "99e199f3847939e8"],
    "WL94A": ["experiments/ecro_mat94r/protocol/ecro_mat94r.json", "5b7c0c8", "d0fd618cb4e1cc9e", "d0fd618cb4e1cc9e"],
    "WL94B": ["experiments/ecro_exp94b/protocol/ecro_exp94b.json", "5c0a6cd", "6b5b10c78d239337", "6b5b10c78d239337"],
    "WL94C": ["experiments/ecro_q94c/protocol/ecro_q94c.json", "6d8b254", "006941ac6cce4741", "006941ac6cce4741"],
    "WL94D": ["experiments/ecro_s94d/protocol/ecro_s94d.json", "a08c9bf", "e0a389463e172bc7", "e0a389463e172bc7"],
    "WL95A": ["experiments/ecro_attr95a/protocol/ecro_attr95a.json", "c152120", "fb136ad2da79fdca", "fb136ad2da79fdca"],
    "WL96A": ["experiments/ecro_util96a/protocol/ecro_util96a.json", "a1718f7", "0a5936504fb8c345", "0a5936504fb8c345"],
}

ARTIFACTS = [  # id, server path, sha256[:16] computed by this audit, produced by, recorded-hash source, match
    ["own_s8601.pt", "~/rest_x92a/results/models/own_s8601.pt", "739e7eb50b25d329", "WL92A (c40226d), written 2026-11-13T02:15:43Z",
     "results/ecro_cf92b/repro.json checkpoints; WL94 pre.json T_CKPT; WL94 worklog l.299", True],
    ["own_s8602.pt", "~/rest_x92a/results/models/own_s8602.pt", "cee9a09a289fca8c", "WL92A (c40226d), 02:15:44Z", "same", True],
    ["ecro_s8601.pt", "~/rest_ecro86/results/models/ecro_s8601.pt", "9d332cd61a7118bd", "WL86", "results/ecro_cf92b/repro.json", True],
    ["ecro_s8602.pt", "~/rest_ecro86/results/models/ecro_s8602.pt", "4d993121e41dd0bb", "WL86", "results/ecro_cf92b/repro.json", True],
    ["CKF_none.npz", "~/rest_x94/results/fields/CKF_none.npz", "3a6238a8a42d1db4",
     "WL94 mat94.fref (e347904), 2 x 4,096 spp, seeds 94301 / 94302, patched pt86a.field_src split=True", "WL96A worklog l.222", True],
    ["masks94.npz", "~/rest_x94/results/pre/masks94.npz (copied to rest_x95a, rest_x96a: identical)", "b9a7d46b1a23d8c4",
     "WL94 mat94.masks (geometry only)", "WL96A worklog l.225", True],
    ["FZ_own_s8601.npz", "~/rest_x94/results/tables/FZ_own_s8601.npz", "44fade056130ec75", "WL94 mat94.ck_tables own:8601", "WL96A worklog l.224", True],
    ["FZ_own_s8602.npz", "~/rest_x94/results/tables/FZ_own_s8602.npz", "03183dd2e6e013dc", "WL94 mat94.ck_tables own:8602", "WL96A worklog l.224", True],
    ["REF_ER_CK_none_CK2_T1_r0.npz", "~/rest_x94/results/renders/ (copy of ~/rest_x93d_ck2/results/renders/)", "89839c42be19acc9",
     "WL93D checkerboard addendum rev. 2 (checker93d.py @ d8d092f), 8,192 spp, seed 86100", "rest_x94/results/pre/reused_refs.txt", True],
    ["REF_ER_CK_none_CK2_T1_r1.npz", "same", "85e8176b0a500b9e", "same, seed 86101", "reused_refs.txt", True],
    ["REF_ER_CK_none_CK2_T1_r2.npz", "same", "a701f32b714b7c2f", "same, seed 86102", "reused_refs.txt", True],
    ["REF_ER_CK_none_CK2_T1_r3.npz", "same", "84fe592778563928", "same, seed 86103", "reused_refs.txt", True],
    ["T95A.npz", "~/rest_x95a/results/tables/T95A.npz (server only)", "d009c57b018e9f39", "WL95A a95.main (c152120)", "WL96A worklog l.223", True],
    ["T96A.npz", "~/rest_x96a/results/tables/T96A.npz = results/ecro_util96a/T96A.npz", "8019cb39d03421dd", "WL96A w96a.tails (a1718f7)", "WL96A worklog", True],
    ["S1_ops.npz", "~/wl95_calc/S1_ops.npz = results/wl95_calc/S1_ops.npz", "8654ae794a9a61f0", "WL95 wl95_calc.py (CPU; inputs rest_x92/results/nmf.npz, rest_x94/results/select94.json, rest_ecro86)", "WL95A worklog T_OPS", True],
    ["scene_ck2.json", "experiments/ecro_local93d/addendum_checker/scene_ck2.json", "977ba122ab3232e7", "WL93D addendum rev. 2 (d8d092f)", "rest_x93d_ck2 chain.status scene=; WL94 pre.json scene_sha256", True],
    ["WL94D audit.json", "~/rest_x94d/results/audit94d/audit.json", "1d03d8432892f5ff", "WL94D s94.audit", "not recorded in WL95A worklog", None],
    ["mat94 run-time source patch", "pt86.Ctx.trace / pt86a.field_src patched text", "57b30a3268542c6f / 14f7367ceebdd013", "mat94.install() regex patch", "pre.json, CKF_none.json, REF json", True],
]

EDGES = {  # consumer: {upstream: [what is consumed]}
    "WL86": {"WL84(ER)": ["fnto_compiled83 closure83_dr / render83 (learned ER closure = material model of every reference and of M64)"]},
    "WL86A": {"WL86": ["checkpoints ecro_s8601/2", "modules pt86, scene86, ecro86, model86, train86"]},
    "WL87": {"WL86": ["fields F_*", "renders EC_*", "final.json", "qualify.json", "stage1.json", "masks86", "modules"], "WL86A": ["fields D_SPLIT_*", "stageB importance_*"]},
    "WL88": {"WL87": ["tables X_E2_T1.npz", "modules quad87 / xport87 / pt87 / fx87"], "WL86": ["modules"]},
    "WL89": {"WL88": ["modules rel88 / tests88 (T_EDIT gate) / analyze88 (post-fix c5e5e2f version) / masks88 / fx88"], "WL87": ["modules xport87, pt87"], "WL86": ["modules"]},
    "WL90": {"WL89": ["w_choice.json (V3; produced before the amendment)", "fields F89_*, ORD_* (pre-amendment run, generator code not touched by the amendment)", "masks", "stage3_89.json (post-amendment economics)", "modules rel89 / analyze89 / econ89"], "WL88": ["modules"]},
    "WL91": {"WL89": ["w_choice.json", "fields F89_*, ORD_*", "masks", "REF_ER_* renders (all pre-amendment artefacts)"], "WL90": ["tables F_W1_T1.npz", "modules gap90 / fx90"]},
    "WL92": {"WL86": ["data (train / val)", "fields F_*", "masks", "models ecro_s860x", "renders", "final.json"], "WL86A": ["pt86a.field_src (amended 1bd7df1 version, split=True)"], "WL87": ["quad87 / xport87 -> M64 (diag92.local_matrices)"]},
    "WL92A": {"WL86": ["training data (sample-identical regeneration of component targets)", "renders (reproduction test)"], "WL92": ["e4_choice.json", "fields D_/W_", "masks E4*", "renders E4", "modules fx92 / diag92 / analyze92"]},
    "WL92B": {"WL92A": ["checkpoints own/sum", "tables I_*", "final92a.json", "fieldeval.json", "fields"], "WL92": ["E4 files", "modules"], "WL86": ["data", "checkpoints"]},
    "WL93": {"WL92A": ["own checkpoints", "exclusion.json", "data *_f2"], "WL92B": ["final92b.json (reconciled)", "e5_candidates.json", "fields", "masks E5k995*", "tables CF_E2_*"], "WL86": ["data"]},
    "WL93A": {"WL93": ["data val_q", "modules model93 / train93 / prep93 / eval93"], "WL92A": ["exclusion.json", "data"], "WL92B": ["e5, fields D_E2_T1, masks"], "WL86": ["data"]},
    "WL93B": {"WL93A": ["models T0 / TQ", "data", "configs.json", "taskA.json"], "WL92B": ["fields", "masks"]},
    "WL93C": {"WL93A": ["data", "configs.json", "taskA.json"], "WL92B": ["tables CF_E2_T1.npz", "fields", "masks"]},
    "WL93D": {"WL92A": ["own checkpoints"], "WL93A": ["T0 checkpoints", "data", "configs"], "WL92B": ["repro.json (checkpoint hashes)", "tables CF_E2_T1.npz", "fields", "masks"], "WL86": ["val data"]},
    "WL93D-addendum": {"WL93D": ["tables I_E2*", "ops93d"], "WL92A": ["own checkpoints"]},
    "WL94": {"WL93D-addendum": ["scene_ck2.json (CK2)", "REF_ER_CK_{none,bx38,bx75}_r0..3 (reused references)", "tables CK_*"], "WL93D": ["tables I_E2_T1.npz", "ops93d"], "WL92A": ["own checkpoints", "exclusion.json"], "WL92B": ["repro.json"], "WL86": ["train data"], "WL93B": ["eval93b.flight"], "WL93": ["train93.load_own"], "WL92": ["diag92.local_matrices (M64)", "fx92"]},
    "WL94A": {"WL94": ["data", "fields", "masks94", "renders", "models", "pop", "select94.json", "tables CD_*, FZ_own", "stageC_s8601.json", "modules mat94 / c94 / an94"], "WL86": ["train data"]},
    "WL94B": {"WL94": ["data", "fields", "masks94", "select94", "FZ_own", "stageB.json"], "WL94A": ["r94 module / assignments"], "WL92A": ["own checkpoints"], "WL86": ["train data"]},
    "WL94C": {"WL94": ["data", "fields", "masks94", "select94", "FZ_own", "stageB.json"], "WL94B": ["audit (path labels)", "b94 module"], "WL94A": ["r94"]},
    "WL94D": {"WL94C": ["models D_s860x", "pop", "renders", "manifest W_s860x", "tables CD_D"], "WL94B": ["audit"], "WL94A": ["tables CD_C", "r_s860x.json"], "WL94": ["data", "fields (CKF_*)", "masks94", "FZ_own", "stageB.json", "an94.W"], "WL92A": ["own checkpoints"]},
    "WL95(design)": {"WL92": ["~/rest_x92/results/nmf.npz (Mq = M64, NMF16 factors)"], "WL94": ["select94.json"], "WL94D": ["quoted sender / floor numbers"], "WL86": ["data"]},
    "WL95A": {"WL94": ["data", "fields CKF_none", "masks94.npz", "REF_ER_CK_none_r0..3 (origin WL93D addendum)", "FZ_own (learned G term)", "modules mat94 / an94"], "WL94D": ["audit.json", "SN_own_s860x.json (reproduction targets)"], "WL95(design)": ["S1_ops.npz", "wl95_calc.json", "wl95_calc2.json", "wl95_calc.py code"], "WL92A": ["own checkpoints"], "WL87": ["quad87.qtable (physical q256)"]},
    "WL96(design)": {"WL95A": ["delivery / bias numbers"], "WL91": ["life91.json timing anchors (C3URX1 edit 124.7 ms, source 75.8 ms; cross-workload, GM1 V3 not CK2)"], "WL90": ["P2plus <= 0.0009"], "WL87": ["0 better / 3 worse / 36 equal count"]},
    "WL96A": {"WL94": ["CKF_none", "masks94.npz", "REF_ER_CK_none_r0..3", "FZ_own (G)", "modules mat94 / an94"], "WL95A": ["T95A.npz floor table", "floor render r0", "masks check"], "WL92A": ["own checkpoint (ECRO-class generator timing)"], "WL87": ["quad87 / xport87 explicit operator, realization 86001 + 960000+d"], "WL92": ["fx92 (E87 config)"]},
}

SHARED = [
    {"id": "SD1", "name": "learned ER closure (WL83/84; fnto_compiled83/closure83_dr.py)", "consumers": "every batch WL86-WL96A", "role": "material model of all references (REF_ER_*, CKF_*) and of M64; errors are measured inside an ER world", "failure_impact": "all fidelity numbers shift together; relative conclusions mostly preserved", "checked_here": "identity only (file blobs equal the stated commits)"},
    {"id": "SD2", "name": "scene86 / pt86 tracer, 4 x 16 bin layout, 1,270 patches", "consumers": "all", "role": "representation and reference tracer", "failure_impact": "all", "checked_here": "never modified after 7786d09 (git log); deployed blobs equal"},
    {"id": "SD3", "name": "state realization seed 86001 (pt86.py:114-127): ONE offset pattern (4,1,K) and ONE 16-point source pattern (2,1,PSRC) broadcast to all patches", "consumers": "WL86 training data and every 'deployed' table of WL86-WL96A; M64 construction", "role": "deployed geometry state, source encoding, quadrature realization", "failure_impact": "every single-draw 'deployed' number is one draw of a correlated estimator; WL95A: deployed box-A delivery 1.21x vs pooled expectation 1.54x", "checked_here": "code read; WL95A T_DEPLOYED_ID quoted"},
    {"id": "SD4", "name": "M64 = quad87.Quad(realization_us(86001), M_ray 65536, seed 87001, M_dense 16384, seed 87002).enc_matrix() (diag92.py:133-139); stored in ~/rest_x92/results/nmf.npz", "consumers": "WL87 (explicit comparator), WL92 (bounds), WL92B / WL93C / WL93D (physical substitution arms C2_*, X_M*), WL93-WL94D (teacher / targets), WL95 (bounds), WL95A (A4), WL96A (sweeps)", "role": "'exact frozen-ER local operator' / teacher / comparator", "failure_impact": "WL95A/WL96 show the binned M64 sender is itself biased on the grazing sharp case (pooled 1.54x, physical q 1.42x); every earlier use of M64 substitution as 'physical truth' is conditional", "checked_here": "construction traced; quad87/xport87 never modified after 60efecb"},
    {"id": "SD5", "name": "pt86a.field_src event split (amended in 1bd7df1; panel tag = material id 1 at the k = 2 vertex) + mat94 run-time regex patch (sha 57b30a32 / 14f7367c)", "consumers": "WL92-WL96A field references incl. CKF_none; WL94+ renders", "role": "event-split field reference (e2_p / e2_o / e3_p / e3_o)", "failure_impact": "all component-level attributions (single vs global, panel vs other)", "checked_here": "diff read; WL94 pre.json identity tests T_ID_TRACE 9e-6, T_ID_FIELD 4e-8, T_REF_REPRO 8e-7 quoted (not re-run)"},
    {"id": "SD6", "name": "OWN checkpoints own_s8601 / own_s8602 (WL92A)", "consumers": "WL92B, WL93 (frozen global), WL93D, WL93D-addendum, WL94-WL94D (frozen arms / init lineage), WL95A (A0, G), WL96A (G, generator timing)", "role": "the only learned operators behind P0-1 and the 'learned' rows of P0-2..4", "failure_impact": "two seeds of one recipe on one dataset; P0-1 magnitudes are seed dependent (energy 0.43 vs 0.89, delivery ratio 0.52 vs 1.63)", "checked_here": "sha256 recomputed, equals three independent records"},
    {"id": "SD7", "name": "CK2 scene (scene_ck2.json, sha 977ba122; one point light grazing the S1 panel)", "consumers": "WL93D-addendum, WL94, WL94A-D (CK2 evaluations), WL95A, WL96A", "role": "the single workload of P0-2, P0-3, P0-4 and of the WL95A passivity / sender numbers", "failure_impact": "origin is a POST HOC QUALITATIVE non-numbered addendum; one geometry, one light; no held-out replication", "checked_here": "hash chain verified (repo = chain.status = pre.json)"},
    {"id": "SD8", "name": "REF_ER_CK_none_CK2_T1_r0..3 (4 x 8,192 spp, seeds 86100-86103)", "consumers": "WL94, WL94A-D, WL95A, WL96A (image E_H reference, noise bounds)", "role": "image reference", "failure_impact": "all ROI E_H values and their noise floors", "checked_here": "sha256 equal in rest_x93d_ck2 and rest_x94 and reused_refs.txt"},
    {"id": "SD9", "name": "CKF_none.npz (2 halves x 4,096 spp)", "consumers": "WL94, WL94D (flight floor), WL95A (delivery reference, n0 0.059 / 0.070), WL96A (tail reference and path-traced prefix)", "role": "field reference", "failure_impact": "P0-2, P0-3, P0-4", "checked_here": "sha256 equals WL96A record"},
    {"id": "SD10", "name": "masks94.npz (ROI_T, T, T_panel_face, S1, DIFF)", "consumers": "WL94-WL96A", "role": "receiver / ROI masks", "failure_impact": "all CK2 region metrics", "checked_here": "sha256 identical in rest_x94 / rest_x95a / rest_x96a"},
    {"id": "SD11", "name": "an94.W (coded 8 x 8, true layout 4 x 16) and an94.eh", "consumers": "W: WL94 (an94.ex table metrics), WL94A, WL94B (b94.py:47), WL94C (via b94), WL94D (s94.py:129, 157, 287-312); eh: WL94-WL96A", "role": "cos dOmega weight / image error metric", "failure_impact": "W mis-weights azimuth halves of each ring (x0.5 / x1.5 in the grazing ring); neutral for azimuth-uniform (Lambert) radiance; biases S1-field metrics, incl. the WL94D SENDER_IMP decision inputs", "checked_here": "defect confirmed from an94.py:19,32 vs ecro86.json bins_z 4 / bins_phi 16 vs scene86.py:15"},
    {"id": "SD12", "name": "WL86 dataset (~/rest_ecro86/data: train 1024, val 64, trainhq; 2.6 GB, server only)", "consumers": "every trained model WL86-WL94D", "role": "training / validation population", "failure_impact": "all learned results share one data distribution (point lights, one room family)", "checked_here": "existence and counts only"},
    {"id": "SD13", "name": "WL89 workload V3 and artefacts; WL91 life91.json timing", "consumers": "WL90, WL91, WL96 (design Amdahl estimate)", "role": "GM1 explicit comparator workload and timing anchors", "failure_impact": "WL96 cost prediction only; WL96A re-measured on CK2", "checked_here": "anchors 124.66 / 75.81 ms verified in life91.json"},
    {"id": "SD14", "name": "server-only storage on LabServer63 (single user account, no second copy)", "consumers": "all P0 inputs (checkpoints, CKF_none, masks, references, T95A, dataset)", "role": "evidence store", "failure_impact": "loss or silent change is undetectable except through worklog-recorded hash prefixes", "checked_here": "all recorded prefixes still match"},
]

WL89 = {
    "timeline_utc": ["03:55:44 CHAIN_START commit 7515a29", "04:07:26 W1 = V3 chosen", "04:35:33 GATE TESTS FAIL -> STOP STOPPED_INVALID_COMPARISON (tests88.T_EDIT selective_equals_full strict digest)",
                     "04:47-05:59 uncommitted debug scripts dbg89.py, dbg89b.py, dbgp.py, dbgs.py in the run dir; audit89 first run had a test bug (audit89_run1_testbug.json), rerun",
                     "06:01:21 code redeployed at 5d6bb5c (POST-FREEZE AMENDMENT 1: analyze89.py, tests89.py, audit89.py, protocol)", "06:01:30 RESUME; finished stages skipped by rc files; S0_tests rerun -> GATE TESTS PASS 06:02:59",
                     "06:36:46 CHAIN_DONE"],
    "what_changed": "the validity predicate (strict SAM.digest over all stored entries -> operational digest over valid samples and operator tensors); generator code rel89.py / run89.py / fx89.py and thresholds untouched (git diff --stat 7515a29 5d6bb5c)",
    "evidence": "results/gm1_relation89/results/amend1/audit89.json: 2 of the stored dbin entries differ, both invalid in both relations; operational digest equal; operator tensors and CPU transport outputs bitwise equal; perturbing 157,538 invalid entries changes nothing; AUDIT_PASS true (read, not re-executed by this audit)",
    "original_preserved": "results/gm1_relation89/results/orig_stop/ (chain.status, tests88.json, tests89.json, A_verdict.log, COMMIT_freeze)",
    "approval": "protocol text cites a human decision of 2026-11-12; not verifiable from repository artefacts",
    "downstream": {
        "WL90": "pre-amendment artefacts (w_choice.json 04:06:42, F89_* / ORD_* 04:34-04:35, masks) + post-amendment stage3_89.json (analyze90.py:169); its premise (common gap of converged explicit relations vs oracle) is a post-amendment WL89 result",
        "WL91": "pre-amendment artefacts (w_choice.json, F89_*, ORD_*, masks, REF_ER_* renders) + WL90 table; own lifecycle timing (life91.json, selective_equals_full True)",
        "WL96(design)": "WL91 life91.json anchors (124.7 / 75.8 ms, 10.6 ms sweep) -> Amdahl 1.09x / 1.16x; cross-workload",
        "WL96A": "no WL89-WL91 quantity consumed; lifecycle re-measured in w96a.cost",
        "WL92-WL95A": "no WL89 artefact consumed (module imports of gm1_explicit87 only)"},
    "classification": "all WL89 comparative results (C3UR vs R0, floor 0.038 vs oracle 0.014, economics) exist only under the amended gate; WL90 / WL91 inherit that condition",
}

P0_CHAINS = {
    "P0-1": {"claim": "The learned S1 sender has large avoidable error, and the deployed learned operator may violate passivity.",
             "supporting": ["WL92 (panel mirror-bin retention 11-14 %)", "WL92A (OWN retention 0.21)", "WL92B (source-side once-scattered stage; label produced by an implemented rule that differs from the written rule)", "WL93D (~80 % wrong vs M64)", "WL94D (sender rel L1 0.82 -> 0.72 with teacher; W94-weighted)", "WL95 (matched-norm: zero-operator level)", "WL95A B1 (rel L1 1.04 / 1.12 vs M64(s16), 1.08 / 1.19 vs q256; albedo_max 2.26 / 3.08)"],
             "measured_quantity": "S1 field rel L1 / energy / peak of a(c^T s16) on 42 panel patches; operator flux-L1; max over patches x incident bins x channels of sum_out A W_T (a95.py:139-152, 300-320)",
             "protocol": "ecro_attr95a (fb136ad2)", "code": "experiments/ecro_attr95a/a95.py @ c152120 (byte-verified)",
             "inputs": ["own_s8601/2.pt", "CK2 scene", "state 86001", "M64", "q256 (quad87.qtable P 256, Mq 1024, seeds 95001/2)", "S1_ops.npz"],
             "reference": "M64(s16) and physical q256 (split-half 0.007)", "decision": "descriptive (no gate on A0); RANK16 label from A1 / A2"},
    "P0-2": {"claim": "The M64/64-bin sender introduces systematic sharp-delivery bias that is not eliminated by relation sampling.",
             "supporting": ["WL95A B2: A4 pooled 64 draws ratio 1.544, abs rel L1 0.546 +- 0.035 vs deployed 1.208 / 0.524", "WL96 (two CPU oracles, design-phase)"],
             "measured_quantity": "exactly-2-event S1-mediated exitance on box A (54 patches), W_T, vs CKF_none e2_p", "protocol": "ecro_attr95a", "code": "a95.py main / an",
             "inputs": ["CKF_none.npz", "masks94.npz", "relation draws rng 950000+d (64)", "M64"], "reference": "CKF_none e2_p mean of 2 halves; n0 0.059", "decision": "FLIGHT_ATTRIBUTION rule (extension to 64 draws by the boundary rule)"},
    "P0-3": {"claim": "Continuous exact-direction and hit-point ER evaluation substantially closes the CK2 delivery error.",
             "supporting": ["WL95A B2: X1 pooled ratio 1.022, abs rel L1 0.111 +- 0.041 (per-draw median 0.617, worst 2.21); ROI E_H 0.0124 vs floor 0.0118 vs A4 pooled 0.0245"],
             "measured_quantity": "same as P0-2 for arms X1c / X1 (a95.py x1_radiance / x1_rays)", "protocol": "ecro_attr95a", "code": "a95.py",
             "inputs": ["CKF_none.npz", "REF_ER_CK_none r0..3", "masks94.npz"], "reference": "CKF_none; REF_ER 4 x 8,192 spp", "decision": "descriptive within FLIGHT_ATTRIBUTION"},
    "P0-4": {"claim": "The learned broad higher-order resolvent has no meaningful accuracy or lifecycle-cost headroom over explicit recurrence on the tested CK2 workload.",
             "supporting": ["WL96A s1.json: paired ROI change 0.00065-0.00082 vs n_img 0.00146 and M_img 0.005", "cost_s1.0.json: U_ZERO 1.038 edit / 1.043 source vs rule 1.25", "WL96 design prediction 1.09x / 1.16x (WL91 anchors)"],
             "measured_quantity": "ROI E_H paired change of explicit e3_o tail vs path-traced tail; lifecycle wall on LabServer63", "protocol": "ecro_util96a (0a593650)", "code": "experiments/ecro_util96a/w96a.py @ a1718f7 (byte-verified; identical to the pre-freeze dev copy)",
             "inputs": ["CKF_none.npz", "REF_ER_CK_none", "masks94", "FZ_own", "T95A.npz", "explicit operator realizations 86001 and 960000+d", "own checkpoint (timing)"],
             "reference": "CKF_none e3 (2 x 4,096 spp); REF_ER", "decision": "S1_NO_HEADROOM by the predeclared rule; Stage 2 not triggered; no learned resolvent was trained (ceiling argument)"},
    "P0-5": {"claim": "The reported negative results justify rejecting the tested ECRO representation mechanisms, but do not establish a general impossibility for reusable Learned Light Transport.",
             "supporting": ["WL86 NOT SUPPORTED", "WL92 STOP_INTERVENTION_G_rep", "WL92A D_STOPPED", "WL92B C (rule mismatch)", "WL93 STOPPED", "WL93A/93C NO_SIGNAL_AT_SMOKE_SCALE", "WL93B SHARING_SIGNAL", "WL93D attribution labels", "WL94 / 94A / 94C INCONCLUSIVE", "WL94B STOP_INVALID_INTERVENTION", "WL94D NO_RELIABLE_IMPROVEMENT", "WL95 PREMATURE", "WL95A", "WL96 NO_VIABLE_LEARNED_MECHANISM (scoped, dissent)", "WL96A"],
             "measured_quantity": "aggregate of the above", "protocol": "n/a", "code": "n/a", "inputs": ["SD3, SD4, SD6, SD7, SD12 shared by nearly all"], "reference": "n/a", "decision": "interpretive"},
}


def main():
    rev = L("code_revision_check.json")
    crlf = L("crlf_check.json")
    static = L("static_deps.json")
    runs = []
    for r in rev:
        e = {k: r[k] for k in ("rundir", "worklog", "own_dir", "candidate_commits", "stated_executable_commit", "n_server_files",
                               "n_identical_to_stated", "n_different", "n_not_in_stated_tree", "server_code_mtime_min", "server_code_mtime_max")}
        e["post_freeze_or_extra_files"] = [{"path": d["path"], "mtime": d["mtime"], "in_commits": d["blob_found_in_commits"]}
                                           for d in r["different"] + r["not_in_stated_tree"] if r["rundir"] not in crlf]
        if r["rundir"] in crlf:
            c = crlf[r["rundir"]]
            e["crlf_deployment"] = True
            e["identical_after_cr_strip_per_commit"] = c["per_commit_identical_after_cr_strip"]
            e["n_compared"] = c["n"]
        runs.append(e)
    out = {
        "audit": "WL97 track MAP", "repo_head": "efcf86a", "generated_by": "experiments/audit97/MAP/build_map.py",
        "runs_code_revision": runs,
        "protocol_hashes": {k: {"path": v[0], "commit": v[1], "sha256_16_lf": v[2], "chain_status_recorded": v[3],
                                "match": (None if v[3] is None else v[2] == v[3])} for k, v in PROTO.items()},
        "artifact_identities": [dict(zip(("id", "path", "sha256_16", "produced_by", "recorded_in", "matches_record"), a)) for a in ARTIFACTS],
        "batch_edges": EDGES,
        "static_code_dependencies": {k: {"module_deps": sorted(v["module_deps"]), "artifact_deps": sorted(v["artifact_deps"]),
                                         "metric_helper_refs": v["metric_helper"], "seed_state_refs": v["seed_86001_and_state"]} for k, v in static.items()},
        "shared_dependencies": SHARED,
        "wl89_episode": WL89,
        "p0_claim_chains": P0_CHAINS,
        "pre_freeze_runs": {
            "WL96A": {"dev": "~/rest_x96adev 16:55-16:58Z (--dev)", "probe": "~/rest_x96aprobe 16:58-17:04Z (full scale: tails, renders, Stage 2 fit, cost)",
                      "freeze_commit_utc": "2026-11-15T17:04:49Z", "chain_start_utc": "17:05:02Z", "chain_done_utc": "17:09:01Z", "results_commit_utc": "17:13:06Z",
                      "protocol_diff_dev_vs_frozen": ["execution (added)", "pre_freeze_exposure (text extended)"], "w96a_py_dev_vs_frozen": "identical",
                      "probe_vs_frozen_T96A_rel_l1_max": 3.4e-9, "probe_vs_frozen_T96A_rel_max": 7.7e-8, "tail_cost_s_probe_vs_frozen": [0.0034209, 0.0033112]},
            "WL95A": {"dev": "~/rest_x95adev 21:00-21:01Z (--dev, protocol 05a92711)", "freeze_commit_utc": "2026-11-14T21:01:38Z", "chain_start_utc": "21:02:21Z",
                      "protocol_diff_dev_vs_frozen": ["contracts.T_OPS wording", "pre_freeze_exposure text"], "a95_py_dev_vs_frozen": "identical"}},
    }
    fn = os.path.join(O, "dependency_map.json")
    json.dump(out, open(fn, "w", encoding="utf-8", newline="\n"), indent=1)
    print("wrote", fn, os.path.getsize(fn), "bytes")


if __name__ == "__main__":
    main()
