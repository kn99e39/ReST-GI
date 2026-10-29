# Worklog58 — Source-independent rule and persistent intrinsic transport state

Batch identity: **intrinsic_state_reuse_v1**. Worklog numbering is chronological only.

## Authorisation and scope

The user explicitly authorised the new bounded batch after the completed WL57 negative result on2026-10-29. One candidate only; Selective_Recompute only, no new branch/worktree, no production/material/reference changes or automatic successor. A fundamental validity-gate failure closes this candidate before training. The separate GPU-training execution improvement proposal is deferred at the user's request.

Candidate design: [Rule/State and dependency contract](../plans/INTRINSIC_STATE_REUSE_V1.md). Executable protocol: `experiments/fnto_reuse58/protocol/reuse_v1.json`. Code: `experiments/fnto_reuse58`. Original WL53–57 code, thresholds and failed evidence are preserved.

## Source grounding

Inspected the current architecture/research-topic/roadmap, FNTO K/C physical ownership contract, WL53–57 physical and learned-reference findings, WL56 complete-domain selective recomputation, WL57 model/geometry/loss implementation and completed result manifests. Starting source29fd693 onSelective_Recompute. This is not an independent rerun of historical physical measurements: those are inherited evidence, explicitly labelled as such.

WL57 established neither complete dynamic/angular quality nor a useful R advantage. Its reusable weights and per-pose rebuilt state do not establish persistent computed-state reuse. WL56's full/selective physical packages agree while the complete physical target still fails in blue. Neither fact is promoted to a new-candidate success.

## Prospective preflight

The first gate isolates finite angular-interface sufficiency before fitting any new weights. The candidate increases the old16-in/64-out interface to64-in/256-out and defines exact dependency-based persistent state. One compact static GGX surface probe is prospectively included, alongside the unchanged homogeneous medium as the intended main fixture. Its inclusion and strict gate are choices for this candidate, not a claim that WL53 volume references support glossy surface physics.

Independent analytic float64 GGX reference,512/1024/2048 projected-angle grids and independent polar quadrature; fixed alpha/angles/source-pair population, optimal output projection and indistinguishable-input lower bounds. Thresholds and stop actions are fixed in the committed protocol. No learned training, oracle expansion or final primary angular measurement ran before protocol/code commit. Local five focused tests passed; these are implementation tests, not physical-fidelity success.

Reports, measurements and final verdicts will be recorded here and in machine-readable result files. Per user instruction, `docs/Architecture.md` contains central design only; no batch report will be appended there.
