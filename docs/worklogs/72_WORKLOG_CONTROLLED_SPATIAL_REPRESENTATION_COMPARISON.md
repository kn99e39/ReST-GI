# Worklog — Controlled Spatial Representation Comparison (2026-11-03)

## Authorisation and scope

On 2026-11-03 the user issued the brief "Worklog 72 — Controlled Spatial Representation Comparison". It authorises one bounded comparative architecture batch on the WL59 intrinsic-K family:

- **Candidate A** keeps W+UV and adds a physically justified generalisation constraint.
- **Candidate B** is a relational, geometry-conditioned shared transport representation.

Both are compared with the preserved WL59/WL71 baselines under one physical contract. The batch includes independent confirmation references on new ports and a previously unscored geometry.

**Preserved:**

- K/C ownership;
- first-exit and ballistic semantics;
- the fixed A/O for the primary comparison;
- 16 incoming / 64 outgoing bins;
- positivity and passivity;
- the thresholds;
- historical checkpoints.

**Not authorised:**

- automatic adoption;
- coupled GI;
- new angular mechanisms or materials;
- coverage increases after results;
- per-port corrections;
- another research hypothesis.

**Parallel work.** The user chose to run the WL57 walker GPU-efficiency rework (WL71C) alongside this batch. WL72 uses a reworked walker only if the rework passes its equivalence gates before WL72 reference generation starts; otherwise it uses the unchanged walker (see the protocol).

**Population status.** WL66/68/71 populations, WL71 fits and WL71A/B results are development evidence here.

**Server note.** At start, LabServer63's GPU was in use by another project's job (`~/TransGI-Reproduction`, `gen_data.py`, about 0.6 GB). This batch does not touch it.

## Execution log

**2026-11-03.** The protocol and code were frozen at 80c9b26 and deployed to LabServer63. `run72.sh` was launched detached. It waited for the TransGI job; the dev smoke started once the GPU had been idle for 10 minutes.

**2026-11-03 11:18 UTC — dev smoke failed at the `ports` stage.**
- The te2 identity gate (bitwise) failed on `node`.
- Every other key of the fresh LabServer63 build is bitwise equal to the archived LabServer69 WL57 geometry.
- In `node`, exactly one of 537,600 elements differs: port 1066, feature 6, archived −5.78e-16 against fresh 0.0. That feature is exactly 0 for 3,578 of 3,840 ports, so this is a cross-machine float32 rounding residue.
- No fit, reference or evaluation had started; the evidence root contains only inputs.
- The failure went unnoticed until 2026-11-04 05:56 UTC, because the wrapper stops on any failure and nothing watched it. The GPU was idle for that interval.

**Amendment 1 (pre-evidence, disclosed, `protocol/cmp72_v1.json` `amendments`).** `node` must now match to |Δ| ≤ 1e-12 absolute; all other keys remain bitwise. The non-bitwise elements and the maximum delta are recorded in `ports.json`. WL72 uses the archived geometry throughout, as before.

The same entry records two more things:
- **Walker decision.** The WL72A engine is not a drop-in, so `~/rest_walker71c/validated.json` is not created and WL72 uses the unchanged WL57 walker.
- **Incident.** During diagnosis, an accidental interactive re-source of `run72.sh` ran idle checks only. It was killed at 05:59 UTC, before any stage started.
