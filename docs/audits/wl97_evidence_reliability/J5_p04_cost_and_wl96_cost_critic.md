# WL97 audit, phase 2 - Track J5: P0-4 cost half and the WL96 cost-critic arithmetic (adversarial adjudication)

Date: 2026-11-16. Adjudicator: independent phase-2 agent (track J5). Repository `Selective_Recompute`, HEAD `efcf86a` + untracked audit files.

Scope: findings D-01, D-02, D-03, D-04, D-06, D-07, D-09, D-10, D-13, E5-1, E5-2, E5-3, E5-4, E5-5, E5-6, E5-7, E5-9, C2-10, F0-06, and the five listed conflicts.
No training, no tuning, no re-timing, no GPU use. Nothing existing was modified. No direction, architecture or successor is proposed.

Method: every cited number was traced to its stored source and recomputed with own code; where two auditors disagreed the raw files of both were read. Two own checks were run (both local, numpy only):

| check | script | output | purpose |
|---|---|---|---|
| J5-A | `experiments/audit97/J5/j5_check.py` | `results/audit97/J5/j5_check.json` | recomputation of every ratio from `cost_s1.0.json`, `life91.json`, `scale91.json`, `econ87.json` and the raw E5 run files (not the E5 summaries) |
| J5-B | `experiments/audit97/J5/j5_dirty.py` | `results/audit97/J5/j5_dirty.json` | independent count (own segment / oriented-box test) of the WL96 cost critic's dirty-pair set |

Re-timing on the GPU was not needed: no contradiction between D and E5 survives once arms and bases are separated (sections 2.1-2.3).

---

## 1. Summary of rulings

| finding | ruling | material | one-line reason |
|---|---|---|---|
| D-01 | UPHELD | no | U values recompute exactly; clock boundary as described |
| E5-1 | UPHELD | no (confirmation) | independent harness reproduces every part within 1.9 ms |
| E5-2 | UPHELD | yes (acceptance result confirmed) | interleaved U(ZERO) 1.02 edit / 1.04 source; criterion 1.25 not met |
| C2-10 | UPHELD | no | same recomputation; slip 1.16 -> 1.14, 1.98 -> 1.79 confirmed |
| D-02 | UPHELD_WITH_MODIFICATION | yes (one worklog sentence, one roadmap clause) | artefact real and now demonstrated in-pipeline by E5; "dominated" = 53-57 % of the deficit; the worklog had already disclosed the q rise and the artefact-free figure |
| E5-3 | UPHELD_WITH_MODIFICATION | yes (same statements) | artefact part upheld; "over-sized stand-in" is a design assumption of the auditor, not a defect of the frozen protocol |
| D-10 | UPHELD | no | protocol predeclared the stand-in; LAM edit is not a design's edit cost |
| D-03 | UPHELD_WITH_MODIFICATION | yes (interpretation only) | convention-bound: yes; but the 1.13 / 1.57 / 1.78 "kernel" values are mixed-basis upper bounds and 2.6 is not a lifecycle; consistent range is 1.02-1.13 wall, 1.06-1.63 work basis |
| E5-4 | UPHELD | yes (interpretation only) | "sweeps are not the bottleneck" is a wall-time statement of this driver |
| D-04 | UPHELD | yes (scope of P0-4) | resolvent-only design; derived full-operator source ceiling 1.24-1.37 (not quality matched, not measured) |
| F0-06 | UPHELD_WITH_MODIFICATION | yes (scope of P0-4) | arithmetic correct; the verdict label and P0-4 are scoped to "resolvent" and "CK2", but not to prefix ownership, N, frame and wall basis |
| D-09 | UPHELD_WITH_MODIFICATION | yes (one worklog sentence) | premise misquoted (orders 1.70x vs q 1.67x); "ceiling rises with N" holds only for source -> frame with a constant frame; reconciled with E5-5 |
| E5-5 | UPHELD_WITH_MODIFICATION | yes (supports the not-triggered extension) | source -> frame ceiling does rise slightly (1.04 -> 1.05), edit falls (1.02 -> 1.01); neither approaches 1.25 on wall to N = 11,430 |
| E5-6 | UPHELD | no | 20 / 33 / 7 / 19-20 tail orders needed (source / edit / source no frame / edit no frame) |
| D-13 | UPHELD | no | selective = full state; no T1 reference; already disclosed by protocol and worklog |
| E5-7 | UPHELD | no | same; the assumption favours the learned side |
| D-06 | UPHELD_WITH_MODIFICATION | yes (one WL96 sentence; no criterion outcome) | 1.16 / 1.98 slip confirmed; "<= 1" is false even inside the critic's own range (C3UR: 1.003); the 11 ms is the whole WL86 operator stage |
| E5-9 | UPHELD | no | same slip; no value changes side of 1.25 |
| D-07 | UPHELD | yes (one WL96 supporting reason, about the explicit comparator) | counts reproduced exactly by own code; linear per-ray rate contradicted by the cited data source |

No finding of this cluster is REFUTED and none is UNRESOLVED. Six are modified, in each case towards a narrower statement than the phase-1 auditor made.

---

## 2. Conflicts

### 2.1 Conflict (1): D-09 versus E5-5 (does the ceiling grow with patch count?)

They measure different arms on different workloads and on different bases, and they do not contradict each other.

| | D-09 | E5-5 |
|---|---|---|
| data | WL91 `scale91.json`, GM1 room, C3UR sender relation, component timings only | own refinement runs on CK2, R0 `XOp` (the WL96A arm), full edit / source events |
| N | 1,270 / 2,802 / 5,080 | 1,270 / 2,802 / 5,080 / 11,430 |
| removable work | 3 of 4 orders at 2.7 ms each (8.0 ms), or 0.75 x the C3UR J4 table | 3 tail orders, 3.2-3.3 ms |
| ceiling type | synthetic: frame (constant 54.5 ms) + q table + orders | measured R0 / ZERO, interleaved, and Amdahl from parts |
| events | source only | source and edit |

Own recomputation (J5-A):

| N | scale91 growth vs N = 1,270 | scale91 synthetic source ceiling, frame 54.5 ms / no frame | E5 raw: U edit measured (Amdahl) | E5 raw: U source measured (Amdahl) | E5 raw: source no frame (Amdahl) |
|---|---|---|---|---|---|
| 1,270 | - | 1.098 / 1.291 | 1.022 (1.023) | 1.045 (1.038) | 1.121 |
| 2,802 | | 1.117 / 1.307 | 1.020 (1.020) | 1.047 (1.043) | 1.121 |
| 5,080 | q 1.67x, one order 1.70x, C3UR J4 table 1.70x, relation update 1.66x, full relation build 4.40x | 1.136 / 1.296 | 1.016 (1.016) | 1.041 (1.046) | 1.116 |
| 11,430 | not in scale91 | - | 1.009 (1.011) | 1.054 (1.054) | 1.106 |

Reading:

1. **The premise of WL96A line 194 is misquoted.** "tail orders growing slower than the q table and relation build" is wrong for the q table: orders grow 1.70x, the q table 1.67x (equal within noise, orders marginally faster). It is true only against the *full* relation build (4.40x), which is not part of any lifecycle event; the per-edit relation *update* grows 1.66x, the same as the orders. In absolute milliseconds the q table does grow more (+16.6 ms vs +1.9 ms per order), which may be what was meant, but the share, not the increment, sets the ratio.
2. **D-09's "ceiling rises with N" is a dilution effect of a constant frame**, not a growing tail share: without the frame the same data are flat (1.29-1.31; tail share of table work 0.225 / 0.235 / 0.229). It concerns source events only.
3. **E5-5 shows the same pattern on the actual WL96A arm:** source -> frame rises slowly (Amdahl 1.038 -> 1.043 -> 1.046 -> 1.054), source without frame is flat to falling (1.12 -> 1.11), and edit -> frame falls (1.022 -> 1.009) because the CPU-side geometry and relation updates grow about linearly (55.5 -> 504 ms from N = 1,270 to 11,430) while the sweeps grow as N^0.32 on wall. E5-5's title ("does not grow") is slightly too strong for source -> frame; its decision impact (criterion not approached) is correct.
4. The levels differ (1.10 vs 1.04) because the C3UR orders cost 2.7 ms each and the R0 `XOp` orders 1.1 ms; D-09's extrapolated "1.25 near N ~ 4e4" is a three-point linear extrapolation on the more expensive arm with eight times the measured range and should not be quoted.

What can be said about scaling:

- Measured (E5, one scene, wall, 512^2 x 64 spp, J = 4, one light): up to N = 11,430 (9x patches) the zero-cost ceiling is 1.01 (edit) and 1.05 (source). The criterion 1.25 is not approached. The WL96A decision not to run the scaling extension is supported after the fact.
- Structural: explicit orders, the q table and a rank-rho apply are all O(N) in work. N alone does not change the tail's share of table work (about 0.23-0.25 on both workloads), so the frame-free, edit-free source ceiling is about 1.30-1.33 at any N on a work basis; with a frame or with edit work it lies below that. On the GPU-kernel basis E5 measures 1.06 -> 1.16 (512^2 x 64 spp frame) and 1.63 -> 1.33 (no frame) from N = 1,270 to 11,430, i.e. both tend to about 1.3.
- Not established: "no scaling hypothesis reaches 1.25x" as a universal. Depth J (20 tail orders would be needed with the frame, 7 without), several lights, lighter frames on a work basis, and an optimised driver were not measured. The sentence is neither supported nor refuted in that generality; in the form "not reached with patch count on wall up to N = 11,430" it is now supported, by evidence WL96A did not have.

### 2.2 Conflict (2): D-02 versus E5-3 (what produced U(LAM, edit) = 0.92-0.93?)

The two auditors describe the same mechanism from two sides; E5 supplies the in-pipeline demonstration that D lacked.

Decomposition of the recorded deficit (J5-A, from `cost_s1.0.json`):

| | LAM24 | LAM64 |
|---|---|---|
| LAM edit - R0 edit | +10.49 ms | +12.37 ms |
| q-table part inflation (identical code; LAM source-event q = 24.3-24.4 ms = R0) | +5.57 ms (53 %) | +7.02 ms (57 %) |
| generator part in the event (GeoState + `factors`) | +8.36 ms | +8.37 ms |
| Lambda apply | +0.21 ms | +0.29 ms |
| tail orders saved | -3.31 ms | -3.31 ms |
| event CPU time / wall | 3.75 | 4.64 (R0: 1.03) |
| U recorded | 0.932 | 0.921 |
| U without the q inflation | 0.967 | 0.964 |
| U without q inflation, generator at its isolated 5.3 ms | 0.987 | 0.984 |

E5 raw event blocks (J5-A read the run files, not the supplement), LAM24 edit, each against R0 measured in the same process:

| block | q-table part | event CPU | U edit |
|---|---|---|---|
| default (run1 / run3, first) | 34.6 / 32.9 ms | 698 / 690 ms | 0.908 / 0.918 |
| numpy cast in GeoState | 25.0 / 24.9 ms | 157 / 155 ms | 0.966 / 0.972 |
| `torch.set_num_threads(1)` | 25.1 / 25.0 ms | 157 / 154 ms | 0.968 / 0.974 |
| default again (same process) | 32.4 / 32.4 ms | 738 / 728 ms | 0.912 / 0.925 |
| `OMP_WAIT_POLICY=passive` (separate processes) | 25.2 / 27.6 ms | 160 / 176 ms | 0.971 / 0.972 |

Ruling on the cause: `model86.GeoState` (`experiments/ecro86/model86.py:54-58`) casts the int16 / uint8 relation tables to int64 with torch on the CPU, which wakes torch's 16-thread OpenMP pool; the workers then spin and slow the single-threaded Python / Dr.Jit tracing of the q table that follows by 5.6-9 ms, and the event burns 0.58-0.74 s of CPU. The explicit arm casts with numpy (`xport87.py:54-56`) and never wakes the pool. The A-B-A design in two processes plus the passive-wait runs is decisive; the effect is a property of the timing harness path, not of generator work. D-02's caution ("plausible but not demonstrated on the real pipeline") is resolved by E5's blocks.

Three corrections to the auditors:

1. **The worklog had disclosed it.** WL96A section C (line 181) reports the q-table rise as "unattributed" and gives "excluding it, LAM edit would still be about 148-150 ms against R0's 144" (= 0.964-0.967), and line 203 lists the CPU time as unexplained. That figure equals E5's artefact-free measurement (0.966-0.974). What the audit adds is the attribution; what was wrong in the worklog is that the headline table, the interpretation ("Any ECRO-class generator makes edits slower (0.92-0.93x)", line 188), the sentence "the generator path runs multi-threaded CPU work" (line 157) and the roadmap entry carry the inflated figure without the caveat.
2. **"Dominated" (D-02) means 53-57 %** of the deficit, not most of it. The remaining 4.9-5.4 ms is real stand-in cost minus the saved tail.
3. **"Over-sized generator stand-in" (E5-3) is not a defect.** The frozen protocol predeclared the stand-in as "the ECRO-class generator (OWN s8601 `model86.factors` on the new geometry state)". E5's trimmed generator (tokens + global P / Q heads, 2.2 ms wall / 0.97 ms GPU) assumes a resolvent that needs no per-bin heads. The Lambda arm's own factors are per (patch, bin) (`P`, `Q` of shape [3, N x 64, rho], `w96a.py:567-573`), and ECRO's P / Q act through the per-bin `c` / `a` heads, so a real generator for that form may need per-bin outputs of the same or larger size (rho 24-64 against R = 16). The resolvent's learned form was never specified (WL95 Gate D), so 2.2 ms and 5.3 ms are both assumptions; neither is "the" generator cost. GeoState's re-upload of state that `XOp` already uploads (2.5-3.3 ms) is a genuine redundancy of the stand-in.

Defensible statement about a geometry-conditioned generator relative to three sweeps (CK2, N = 1,270, this GPU):

- Fully regenerated on every edit, the predeclared ECRO-class stand-in costs 4.3-5.3 ms isolated (4.2 ms GPU time) plus 2.5-3.3 ms state upload, against 3.2-3.3 ms wall (1.6 ms GPU time) for the three sweeps it would replace. **For the stand-in as predeclared, "the generator costs more than the sweeps" is correct on wall and on GPU time.**
- Its measured effect on the edit lifecycle, with the timing artefact removed, is **0.97x (0.966-0.974, five in-event measurements)**, not 0.92-0.93x. With a P / Q-only generator and shared state it would be 0.98-1.01x (derived, not measured in an event). Both are parity within the 1-2 % run-to-run spread of the edit lifecycle.
- "Any ECRO-class generator makes edits slower" is not supported as a universal; "no ECRO-class generator makes this edit lifecycle measurably faster" is supported.
- Scaling (E5, isolated components): full `factors` grows with exponent 0.98 in N (4.3 -> 37.5 ms at N = 11,430; GeoState 3.3 -> 52 ms) against 0.32 for the sweeps on wall (3.2 -> 6.5 ms). A full per-edit regeneration therefore loses ground at larger N in this implementation. The stand-in is not selective (the explicit arm re-traces 7.7 % of the rays); selective regeneration was not tested.
- None of this touches the frozen criterion: U < 1.25 in every variant.

### 2.3 Conflict (3): D-03 versus E5-2 / E5-4 (which conventions are defensible; how far does "nothing to amortise" reach?)

Both are right about their own object: the criterion outcome is robust on wall (E5-2), and its generalisation to "the transport work" is not (D-03, E5-4). D-03's table needs one correction: its "GPU kernel" rows keep the torch sweeps at their synchronised *wall* time while moving the Dr.Jit parts to kernel time ("upper bound of their GPU time", as its own note says). E5 profiled the sweeps: 1.58 ms GPU against 3.17-3.3 ms wall. On one consistent basis (J5-A, E5 run1 inputs):

| convention (source update, zero-cost resolvent, N = 1,270) | ceiling | status |
|---|---|---|
| wall, frame 512^2 x 64 spp (frozen, predeclared) | 1.038-1.045 measured; Amdahl 1.038 | the acceptance convention |
| wall, any lighter frame (512^2 or 256^2, 16 / 4 / 1 spp) | 1.047-1.053 | defensible; the render has about 35 ms of fixed per-call overhead |
| wall, no frame | 1.12-1.13 | defensible as "table update only"; WL96A reported it (1.12) |
| GPU work, frame 512^2 x 64 spp | 1.06 | consistent basis (E5: 1.063) |
| GPU work, 512^2 x 16 / 4 spp | 1.20 / 1.40 | idealised; not a realised lifecycle |
| GPU work, no frame | 1.63 | idealised; not a realised lifecycle |
| mixed basis as in D-03 (Dr.Jit at kernel time, sweeps at wall): frame 64 spp / no frame | 1.12 / 1.85 (D: 1.13 / 1.78) | not defensible: overstates the tail by about 2x |
| operator applications only (D-03: 2.61) | 2.6 | not a lifecycle; equals (1 prefix + 3 tail) / prefix by construction |

Edit update: wall 1.02 (frame) / 1.04 (no frame); GPU work 1.05-1.06 (64-spp frame), 1.09-1.19 (16 spp), 1.14-1.61 (no frame) depending on whether the CPU geometry work is charged.

Which conventions are defensible for an architecture decision about a learned resolvent:

1. **The frozen convention** (synchronised wall, edit -> frame and source -> frame, 512^2 x 64 spp) is defensible as the *acceptance* convention: it was predeclared, it is the WL87 / WL91 convention, and a user-visible update includes a frame. Under it the outcome is **VERIFIED as measured**: reproduced by an independent harness in three processes with interleaved arms (U(ZERO) 1.021-1.024 edit, 1.040-1.045 source; every LAM variant <= 1.04), holding to N = 11,430. The original U(ZERO, edit) = 1.038 is inflated by arm drift (the ZERO arm gained 5.3 ms while 3.3 ms were removed); 1.02 is the better value. Criterion 1.25 not met; margin >= 0.2.
2. **Wall without frame** is defensible as a descriptive companion. It gives 1.04 / 1.12-1.13 and does not change the outcome.
3. **A consistent GPU-work basis** is defensible only as a *sensitivity analysis* answering "is the result a property of the transport work or of this driver?". It is not a realised lifecycle: it assumes a driver with no Python / tracing / numpy overhead that nobody has built, for the explicit and the learned side alike. On that basis the outcome is unchanged whenever the 64-spp frame is in the denominator (1.06) and reverses only for source updates with a frame lighter than about 512^2 x 16 spp, or none (1.2-1.63).
4. Mixed bases and operator-only ratios are not defensible as decision inputs.

How far the interpretation reaches:

- "COST criterion not met" (line 160): VERIFIED as measured.
- "No lifecycle headroom" (line 188) and "nothing to amortise in cost, because the sweeps are not the lifecycle's bottleneck" (line 205): hold for **every wall-time convention** (<= 1.13) and for **every convention that includes the 64-spp frame** (<= 1.06, wall or GPU work), for a resolvent that replaces three orders behind an explicit q table and first order, at N = 1,270-11,430, J = 4, one light, one scene, one implementation. They do not hold as a property of higher-order transport work: 69-81 % of each lifecycle is CPU-side driver overhead (GPU kernel time is 17.7 % of the edit wall and 27.4 % of the source wall; the q table is 7.6 % kernel), and on a work basis the three orders are 39 % of the source table update.
- One inconsistency of convention between the two worklogs must be recorded. WL96 (line 151) discounts the only measured learned cost advantage (WL87 source-only update, 3.5-3.9x on wall, recomputed) as "overhead-confounded" because the explicit q table is 42-54 ms wall against 2 ms GPU. WL96A then bounds the learned resolvent with the same overhead in the denominator. Each statement is factually correct; together they apply the work basis where it weakens the learned side and the wall basis where it weakens the learned side. On one basis throughout: **wall** - resolvent ceiling 1.04, full-operator source ceiling about 1.24-1.37 (section 2.4); **GPU work with the 64-spp frame** - resolvent 1.06, full operator about 1.1. Neither basis supports "no learned cost advantage of any kind"; both support "no resolvent-only advantage >= 1.25 with a 64-spp frame".

### 2.4 Conflict (4): D-04 / F0-06 (resolvent-only scope; is any sentence written as if it covered learned transport in general?)

The scope finding is upheld. By protocol (`ecro_util96a.json`, `stage_3.arms`) the LAM and ZERO arms keep the explicit q table (24-26 ms wall) and one explicit order; only orders >= 3 events are replaced. Derived, not measured (J5-A): a source path that needed neither q table nor sweeps would bound at 88.5 / 58.8 = 1.51x; with ECRO's recorded source encoding (5.1-5.4 ms, WL87) and the isolated apply (0.77 ms) 1.36-1.37x; with WL87's recorded application including the GeoState rebuild 1.24-1.28x. These figures are not quality matched (the same WL87 found the explicit arm at least as accurate) and inherit the q table's 93 % driver overhead, so they are not evidence *for* a learned advantage either. They show only that the WL96A ceiling is a statement about one role.

Sentence audit:

| location | sentence | assessment |
|---|---|---|
| WL96A title line / verdict (209) | "LEARNED_RESOLVENT_UTILITY = NO_MEANINGFUL_HEADROOM (CK2 `none`; ...)" | scoped to the resolvent and CK2; silent on prefix ownership, N, frame and wall basis. Acceptable as a label if read with section B4. |
| WL96A 188 | "Any ECRO-class generator makes edits slower (0.92-0.93x)." | over-general ("any") and artefact-inflated. |
| WL96A 192 | "With no headroom over explicit, no representation can buy anything on this workload." | over-general for cost: true for the resolvent role only; a representation owning the q table is outside the measurement. |
| WL96A 194 | "... so no scaling hypothesis reaches 1.25x." | premise misquoted; universal not established (section 2.1). |
| WL96A 205 | "Why the result does not justify learning. ... nothing to amortise in cost, because the sweeps are not the lifecycle's bottleneck, and a geometry-conditioned generator costs more than the sweeps." | body is about the resolvent; the heading is general. Wall-basis statement; generator clause true for the predeclared stand-in only. |
| WL96A 212 | "its dissent ... is now answered *for the broad-last resolvent*" | correctly scoped. |
| WL96 144-149 | Amdahl ceilings; "With an ECRO-class generator (about 11 ms per edit), the per-edit ratio is <= 1." | resolvent-only (in fact all sweeps removed, which is generous); slip and unlike quantities (section 2.5). |
| WL96 151 | "The only measured learned advantage is overhead-confounded." | concerns the full operator; factually correct, convention-inconsistent with the ceiling (section 2.3). |
| WL96 177 | "Learned resolvent (K2). ... Measured same-room economics cap its benefit below 1.25x lifecycle" | correctly scoped to the resolvent; "cap" holds for frame-inclusive conventions. |
| WL96 178 | "[INF] Conclusion. Learning fails where explicit is expensive ... and is redundant where explicit is cheap. No candidate makes a supportable claim that the learned method is preferable ..." | **the one sentence written in general form.** Labelled [INF] and followed by "No candidate", but the first clause reads as a statement about learning. Its cost support is the resolvent-only ceiling plus the discounted WL87 advantage; one of its "explicit is expensive" supports (60 % dirty pairs) is invalid (section 2.5). |
| WL96 185 | "The single falsifiable learned element (K2's broad resolvent) ... is predicted decorative by verified same-room measurements." | scoped to the resolvent; "decorative" is a wall / frame-inclusive statement. |
| `critic_cost.md` 1.4, 2.2, 2.3 | "Learning fails where explicit is expensive and is redundant where explicit is cheap."; "a guaranteed loss"; "the learned part is decorative" | supporting material; stronger wording than the worklog adopted. |
| Roadmap 155 | "predicted decorative: a zero-cost resolvent caps lifecycle gains at 1.09x edit->frame / 1.16x source->frame (WL91 anchors)" | scoped; 1.16 should read 1.14. |
| Roadmap 156 | "a rank-24/64 resolvent with an ECRO-class generator 0.92-0.93x edit / 1.04x source"; "on CK2 there is neither accuracy nor lifecycle headroom for a learned >=3-event resolvent behind a physical prefix" | second clause correctly conditional (as F0-06 says); first carries the artefact-inflated figure. |

No sentence in `docs/Architecture.md`, `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md` or `docs/RESEARCH_CENTRIC_TOPIC.md` cites these cost numbers. F0-06's remark that "the verdict label and P0-4 wording are not [conditional]" is modified: both are scoped to the resolvent and to CK2; what they omit is the prefix-ownership, size and convention conditions.

### 2.5 Conflict (5): the WL96 cost critic (D-06, D-07, E5-9, C2-10)

**(a) Amdahl table** (`critic_cost.md` section 2.2, WL96 lines 144-148; recomputed from `life91.json`, C3URX1 arm):

| row | critic | recomputed | note |
|---|---|---|---|
| edit -> frame | 1.09 | 124.66 / (124.66 - 10.57) = 1.093 | correct |
| edit without render | 1.18 | 69.58 / 59.00 = 1.179 | correct |
| source -> frame | 1.16 | with the source event's own sweeps (9.40 ms): 1.142; 1.162 only with the edit event's 10.57 ms | slip |
| source without render | 1.98 | own sweeps: 21.34 / 11.93 = 1.788; 1.983 with the edit event's sweeps | slip |

The slip is confirmed by three auditors and by J5-A. No value changes side of 1.25. The anchors are the non-adopted C3URX1 arm with *all* J = 4 orders removed; R0 gives 1.040 / 1.070 and C3UR 1.090 / 1.135, so the choice is the generous one. Conclusions that depended on the slip: none.

**(b) "With a real generator of about 11 ms per edit (ECRO-class) instead of 5.6-11.6 ms of sweeps, the per-edit ratio is <= 1, a guaranteed loss"** (critic 2.2; WL96 line 149 keeps "<= 1"):

- The 11 ms is WL86's "operator" stage = `EC.respond` (`ecro86.py:82-87`): GeoState construction (with the CPU cast of section 2.2), `factors`, `apply` and the download - the whole learned replacement of all four orders including the first flight, not a generator. WL96A later isolated `factors` at 5.3 ms.
- The statement fails inside its own range (J5-A): replacing the sweeps by 11 ms gives 0.964 (R0, 5.6 ms sweeps), 0.997 (C3URX1, 10.6 ms) and **1.003 (C3UR, 11.4 ms)**; with 5.3 ms: 1.002 / 1.044 / 1.046.
- Corrected: parity within timing noise (0.96-1.05 depending on arm and on what is charged). "<= 1" and "guaranteed loss" are not supported. "No gain approaching 1.25" is.
- Conclusions that depended on it: the cost half of "predicted decorative" (WL96 line 185, roadmap 155) and the "redundant where explicit is cheap" clause. Both survive in the weaker form "no demonstrated gain"; neither survives as "loss".

**(c) "60 % dirty pairs => about 5.5 M re-tests => about 60 ms => 136-140 ms per edit, about 1.8x ECRO"** (critic 1.3; WL96 lines 114 and 153):

J5-B reproduces D's counts exactly with independently written geometry code (front-facing pairs 9,093,120):

| set (box A shifted -0.3 m in y) | pairs | share of front-facing |
|---|---|---|
| crossing box A at the old or the new pose (the critic's "dirty"), static receivers | 5,391,278 | 59.3 % (61.1 % with receivers on A) |
| crossing at both poses (blocked before and after: visibility cannot change) | 3,586,239 | 39.4 % |
| crossing at exactly one pose | 1,805,039 | 19.9 % (21.6 % with receivers on A) |
| of which old-only (may have become visible: a visibility ray is needed) + receivers on A | 709,741 + 163,584 | 9.6 % |

- The count itself is correct as "pairs touched by the swept volume", and it uses the same old-or-new rule as the project's own relation update (`ecro86.dirty_rays`). As a number of visibility *re-tests* it is an upper bound for a naive implementation: two thirds of it are pairs blocked at both poses, and new-only pairs are decided by the box test alone. 0.87-1.97 M lies inside designer d2's "1-2 M rays" estimate that the critic rejected.
- The millisecond conversion assumes a linear rate (1.05 M rays per 11.5 ms) from one WL91 point. The same file contradicts it: the X1 source evaluation is flat at 11.1-14.0 ms from 0.26 M to 2.1 M landed rays (slope 0.22 ns per ray, r = 0.13). The cost of 5.5 M (or 2 M) pair tests was never measured; "about 60 ms" is unsupported, and the alternative figures (9.6-21.6 ms at the critic's own rate; about 13 ms from the flat fit) are extrapolations too.
- Corrected statement: for this edit 10-22 % of the front-facing S1 pairs need a new visibility decision (0.9-2.0 M); the per-edit cost is unmeasured and by the critic's own rate would be 1.1-1.3x ECRO's edit, not 1.8x.
- Conclusions that depended on it: the critic's prediction that the designers' edit budgets (d2 F2e, d4 FA5, d3 FC) fail "on their own favourable workload", the first item of "explicit carriers carry real costs that the designers underestimated" (WL96 line 153), the DQ row of the comparator table (line 114) and part of the "learning fails where explicit is expensive" clause. All concern the cost of the *explicit comparator* carriers. The correction makes the explicit carrier cheaper, so it does not weaken any conclusion against a learned mechanism and does not help a learned mechanism; it removes one stated reason and one predicted comparator failure. The alpha^-4 pair scaling and the rotating-sharp-mover case (all pairs of the mover invalid) are untouched.

---

## 3. Remaining findings (short)

- **D-01, E5-1, E5-2, C2-10 (confirmations).** U recomputed from `cost_s1.0.json` equals the recorded values to four digits (J5-A). E5's independent harness (no import of `w96a.py`) reproduces the lifecycles within +1.3 to +3.3 % and every part within 1.9 ms; dirty rays 49,936 identical. No timed event compiled. Thin original design (6 edits, sequential arms, medians) is a fair note and is covered by E5's 12-edit interleaved repeats.
- **D-10.** Upheld. The LAM arm models the apply cost and the source events correctly; its edit number is a full-regeneration stand-in with the section 2.2 artefact. The protocol says so; the worklog headline does not.
- **E5-6.** Upheld (J5-A: 20 / 33 / 7 / 19 tail orders needed at 1.10 ms per order). The with-frame conclusion is insensitive to plausible depth; the frame-free source convention is not.
- **D-13, E5-7.** Upheld, not material, already disclosed (protocol `stage_3.edit`; worklog lines 127, 202). The selective update equals a full rebuild record for record (E5 in 7 / 7 runs up to N = 11,430; D on frozen code); U is a quality-free ratio that assumes the learned arm reaches R0's T1 quality at no extra per-edit work, which favours the learned side. It cannot be cited as quality-matched dynamic-GI economics.
- **E5-4.** Upheld (section 2.3). Its own caveat is the right one: the kernel basis is an idealisation.

---

## 4. P0 verdicts decided by this cluster

### P0-4, cost half - "the learned broad higher-order resolvent has no meaningful lifecycle-cost headroom over explicit recurrence on the tested CK2 workload"

**CONDITIONALLY_VALID.** Component verdicts:

| component | verdict |
|---|---|
| Frozen acceptance result: COST criterion (U >= 1.25) not met; scaling extension not triggered | **VERIFIED** (as measured; reproduced independently; robust to arm drift, interleaving, three processes, N up to 11,430) |
| "No lifecycle headroom" / "nothing to amortise in cost" as an interpretation | **CONDITIONALLY_VALID** |
| "Any ECRO-class generator makes edits slower (0.92-0.93x)" | **IMPLEMENTATION_DEFECT** (timing artefact; corrected 0.97x for the predeclared stand-in, about parity) |
| "no scaling hypothesis reaches 1.25x" | **UNTESTED_INFERENCE** as written (premise misquoted); supported after the fact for patch count on wall up to N = 11,430 |
| Extension to learned operators that own the q table / prefix, or to learned transport in general | **UNTESTED_INFERENCE** |

Exact restrictions under which the claim holds: (i) the learned element replaces only the three >= 3-event orders and still pays for the explicit q table, the first order, the relation update and the frame; (ii) lifecycle = synchronised wall time of this Python / Dr.Jit / torch prototype, or any basis that keeps the 512^2 x 64-spp frame in the denominator; (iii) CK2, one point light, J = 4 orders, N = 1,270 (to 11,430 by the audit's own runs); (iv) one timing-only rigid edit of box A with no T1 reference, so the ratio is quality-free; (v) one GPU, one implementation. Under (i)-(v) the ceiling is 1.02 (edit) / 1.04 (source) and at most 1.13 without the frame. Outside them: on a GPU-work basis without a frame the source ceiling is 1.6 (idealised); a full learned operator has a derived, quality-unmatched source ceiling of 1.24-1.37 on wall; depth, lights and optimised drivers are unmeasured.

### P0-5, cost-argument contribution only (this cluster does not decide P0-5 as a whole)

**CONDITIONALLY_VALID.** The cost evidence justifies: "no cost advantage >= 1.25x has been shown, or is available under frame-inclusive conventions, for a resolvent-only learned element on the tested workloads". It does not establish that learned transport lacks lifecycle-cost headroom: the bound is role-specific (section 2.4), convention-bound (section 2.3), and two of the WL96 cost critic's supporting statements are invalid comparisons ("<= 1 / guaranteed loss"; "60 % dirty pairs, 1.8x ECRO"). The second half of P0-5 ("do not establish a general impossibility") is consistent with everything in this cluster.

---

## 5. Affected worklog statements

Format: original claim -> defect or limit -> corrected interpretation -> known dependents -> action. Worklogs are append-only; "QUALIFY" / "RETRACT" mean a dated correction in a later record, never an edit of the worklog.

1. **WL96A 133-134, 188; roadmap 156** - "LAM24 / LAM64 U edit 0.93 / 0.92"; "Any ECRO-class generator makes edits slower (0.92-0.93x)".
   -> 53-57 % of the deficit is an OpenMP spin-wait artefact triggered by a torch CPU cast in the stand-in's state upload; the stand-in is a full, non-selective regeneration of ECRO's own `factors`.
   -> With the artefact removed the predeclared stand-in gives 0.97x (0.966-0.974); a P / Q-only generator about 0.98-1.01x (derived). Parity, not a 7-8 % loss. The worklog's own section C figure (148-150 ms) was already the right one.
   -> WL96A 205 ("a geometry-conditioned generator costs more than the sweeps"), roadmap 156, the coordinating session's memory notes for WL96A.
   -> **QUALIFY** (replace the number; drop "any").

2. **WL96A 157, 203** - "the generator path runs multi-threaded CPU work"; "CPU time on the generator path ... is unexplained. It does not change the ordering."
   -> Now explained: the CPU time is spinning OpenMP workers, not generator work; it does change the magnitude (0.92 -> 0.97) though not the criterion outcome.
   -> attribution only.
   -> **QUALIFY**.

3. **WL96A 135, 188** - "ZERO ... U edit 1.038"; "at most 1.04x".
   -> The edit figure is inflated by arm drift (5.3 ms gained, 3.3 ms removed); interleaved repeats and Amdahl give 1.02. Source 1.04 stands.
   -> none (the error favours the learned side).
   -> **PRESERVE** (note 1.02).

4. **WL96A 160-161** - "COST criterion (U >= 1.25) not met. The scaling extension ... was not triggered."
   -> none.
   -> stands as measured; additionally supported to N = 11,430.
   -> `final.json` verdict NO_MEANINGFUL_HEADROOM (cost half).
   -> **PRESERVE**.

5. **WL96A 194** - "WL91's GM1 scaling shows the tail orders growing slower than the q table and relation build ..., so no scaling hypothesis reaches 1.25x."
   -> orders 1.70x vs q table 1.67x (equal); slower only than the full relation build, which no lifecycle event contains; the universal is not established.
   -> "Not measured in WL96A. scale91 shows orders, q table and relation update growing alike. Audit runs on CK2 show U(ZERO) 1.02 -> 1.01 (edit) and 1.04 -> 1.05 (source) from N = 1,270 to 11,430 on wall. Depth, lights and work-basis conventions are untested."
   -> the decision not to run the scaling extension (unaffected).
   -> **RETRACT** the sentence as written; **REVALIDATE** is not required for the criterion (already covered by E5's runs).

6. **WL96A 188, 205** - "No lifecycle headroom"; "nothing to amortise in cost, because the sweeps are not the lifecycle's bottleneck".
   -> wall-time statement of an overhead-dominated prototype (GPU kernel 17.7 % / 27.4 % of the edit / source wall) for a resolvent-only role; on a work basis the three orders are 39 % of the source table update.
   -> holds for all wall conventions (<= 1.13) and all conventions with the 64-spp frame (<= 1.06); not a property of higher-order transport work; not a statement about learned operators that own the q table.
   -> WL96A verdict text, roadmap 156 (already conditional: "behind a physical prefix"), P0-4 wording.
   -> **QUALIFY**.

7. **WL96A 192** - "With no headroom over explicit, no representation can buy anything on this workload."
   -> over-general for cost: only the resolvent role was bounded.
   -> "no representation in the resolvent role".
   -> none known.
   -> **QUALIFY**.

8. **WL96A 162, 202** - "Without the frame ..., the ceilings are 1.05x edit and 1.12x source."
   -> none (recomputed 1.04 / 1.12; E5 1.04 / 1.13).
   -> stands; wall basis.
   -> **PRESERVE**.

9. **WL96 146, 148; `critic_cost.md` 2.2; roadmap 155** - "source->frame 75.8 / 65.2 = 1.16x"; "source without render (1.98x)".
   -> the edit event's sweep time (10.57 ms) was subtracted from the source lifecycle; the source event's own is 9.40 ms.
   -> 1.14x and 1.79x. Same side of 1.25 in both cases.
   -> WL96A 189 ("1.09x / 1.16x"), roadmap 155.
   -> **QUALIFY** (numerical correction; no conclusion changes).

10. **WL96 149; `critic_cost.md` 2.2** - "With an ECRO-class generator (about 11 ms per edit), the per-edit ratio is <= 1" / "a guaranteed loss".
    -> 11 ms is the whole WL86 operator stage (state upload with the CPU cast, factors, apply, download); the claim fails inside its own range (C3UR sweeps 11.4 ms: 1.003).
    -> parity within noise (0.96-1.05 by arm and by what is charged); no gain approaching 1.25.
    -> "predicted decorative" (WL96 185, roadmap 155), cost half only; survives as "no demonstrated gain".
    -> **RETRACT** "guaranteed loss" / "<= 1"; **QUALIFY** "decorative".

11. **WL96 114, 153; `critic_cost.md` 1.3** - "a 0.3 m box-A shift dirties 60 % (about 5.5 M re-tests, about 60 ms)"; "about 136-140 ms per edit, about 1.8x ECRO"; designers' edit budgets "predicted to fail".
    -> the 60 % is the old-or-new union (39 % of the pairs are blocked at both poses and need no re-test); the linear ms rate is contradicted by the cited WL91 data (flat 11-14 ms for 0.26-2.1 M rays).
    -> 10-22 % of the pairs (0.9-2.0 M) need a new visibility decision, inside designer d2's estimate; per-edit cost unmeasured, 1.1-1.3x ECRO at the critic's own rate.
    -> WL96 "explicit carriers carry real costs that the designers underestimated" (first item), the "explicit is expensive" half of line 178. Concerns the explicit comparator; no effect on the learned-mechanism verdict other than removing one stated reason.
    -> **RETRACT** the prediction and the 136-140 ms figure; **PRESERVE** the count as an upper bound labelled as such.

12. **WL96 151** - "The only measured learned advantage is overhead-confounded."
    -> correct as a fact (q table 42-54 ms wall, 2 ms GPU), but it applies the work basis to the learned advantage while the ceilings in the same section and in WL96A use the wall basis.
    -> on one basis: wall - full-operator source advantage stands (3.5-3.9x table-only as measured in WL87; about 1.24-1.37x with a frame, derived, not quality matched) and the resolvent ceiling is 1.04; GPU work with the 64-spp frame - about 1.1 and 1.06.
    -> WL96 178, the K2 paragraph (177).
    -> **QUALIFY**.

13. **WL96 178; `critic_cost.md` 1.4** - "[INF] Learning fails where explicit is expensive ... and is redundant where explicit is cheap."
    -> general wording; cost support is resolvent-only, convention-bound, and partly rests on items 10-11.
    -> an inference about the four candidate designs of WL96, not about learned transport.
    -> WL96 verdict rationale; living-document summaries.
    -> **QUALIFY** (scope to the candidates; cost half conditional).

14. **WL96 177** - "Measured same-room economics cap its benefit below 1.25x lifecycle."
    -> holds for the resolvent under frame-inclusive conventions (1.09 / 1.14 on C3URX1, 1.04 / 1.07 on R0; confirmed on CK2).
    -> stands with the restrictions of section 4.
    -> **PRESERVE** (with item 9's number).

15. **WL91 lifecycle and scaling tables** (`life91.json`, `scale91.json`) - reproduce exactly; no defect found in this cluster. -> **PRESERVE**.

---

## 6. Unresolved or not decided here

- The per-edit cost of a deterministic sender-cell x receiver-point quadrature update (DQ / CIS) was never measured; both the critic's 60 ms and the audit's 10-22 ms are extrapolations.
- The cost of a learned resolvent's generator is undefined while its learned form is unspecified (WL95 Gate D): 2.2 ms (P / Q only), 5.3 ms (`factors`) and 8 ms (`factors` + state upload) are all stand-ins.
- No quality-matched cost comparison exists for any edit in this cluster (no T1 reference on CK2); all U values are quality-free.
- Depth (J > 4), several lights, and an optimised (low-overhead) driver are unmeasured for both the explicit and the learned side.
- The 1.24-1.37x full-operator source ceiling is derived from component medians of two different batches (WL87, WL96A) and is not a measurement.
- The mechanism of the OpenMP slowdown is established by intervention (thread count, cast path, wait policy), not by profiling the q-table code itself.

## 7. Commands

```
python experiments/audit97/_tools/show.py D-01 D-02 D-03 D-04 D-06 D-07 D-09 D-10 D-13 E5-1 E5-2 E5-3 E5-4 E5-5 E5-6 E5-7 E5-9 C2-10 F0-06
python experiments/audit97/J5/j5_check.py     # -> results/audit97/J5/j5_check.json
python experiments/audit97/J5/j5_dirty.py     # -> results/audit97/J5/j5_dirty.json  (about 5 s)
```

No server command was issued by this track.
