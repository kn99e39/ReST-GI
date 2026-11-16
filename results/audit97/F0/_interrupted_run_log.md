# Track F0: action log of the interrupted first attempt (auto-extracted by the coordinator; UNVERIFIED; commands and truncated outputs only)

RUN [List worklogs and document sizes] ls docs/worklogs | sed -n '80,200p' && wc -c docs/worklogs/8[6-9]* docs/worklogs/9* && ls docs/audits 2>/dev/null; ls docs/audits/wl97_evidence_reliability 2>/dev/null; wc -l docs/RESEARCH_ROADMAP.md docs/Architecture.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md docs/RESEARCH_CENTRIC_TOPIC.md
  -> 76_WORKLOG_FIRST_INTERACTION_ANCHORED_TRANSPORT.md
77A_WORKLOG_MULTI_AGENT_ARCHITECTURE_DISCOVERY.md
77_WORKLOG_FIA_ANGULAR_EXPRESSIVITY.md
78_WORKLOG_SOURCE_IDENTITY_LEAKAGE.md
79_WORKLOG_ANGULAR_FRAME_EQUIVARIANCE.md
80A_WORKLOG_MULTI_AGENT_ARCHITECTURE_THOUGHT_EXPERIMENT.md
80B_WORKLOG_C1_ARCHITE ...[cut]... _SUPERVISED_SMOKE.md
 24307 docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
 19537 docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md
 68930 docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md
 21674 docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md
 24734 docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md
669824 total
   171 docs/RESEARCH_ROADMAP.md
   116 docs/Architecture.md
    78 docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md
    58 docs/RESEARCH_CENTRIC_TOPIC.md
   423 total
RUN [List worklogs and document sizes] cd /c/Projects/ReST-GI && ls docs/worklogs | sed -n '80,200p' && wc -c docs/worklogs/8[6-9]* docs/worklogs/9* && ls docs/audits 2>/dev/null; ls docs/audits/wl97_evidence_reliability 2>/dev/null; wc -l docs/RESEARCH_ROADMAP.md docs/Architecture.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md docs/RESEARCH_CENTRIC_TOPIC.md
RUN [Read baseline roles strategy doc] cat docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md | cut -c1-2500
RUN [Read Architecture doc] cat docs/Architecture.md | cut -c1-3000
  -> # Baseline Roles and Evidence Strategy

**Current as of:** 2026-11-02. **Role:** baseline roles, comparison controls, evidence limits and evaluation accounting. [Architecture](Architecture.md) owns candidate dispositions; the [roadmap](RESEARCH_ROADMAP.md) owns current bounded scope. This document ...[cut]... vidence, its bounded negative learned result and the WL56 physical quality floor remain preserved. [WL57](worklogs/57_WORKLOG_LEARNED_INTRINSIC_K_RULE_STATE.md) owns these distinctions; abstract Rule–State superiority remains INCONCLUSIVE.

The [intrinsic-state reuse preflight](worklogs/58_WORKLOG_PERSISTENT_INTRINSIC_STATE_REUSE.md) stopped on its finite angular representation before training or state-update implementation. Treat that as candidate admission evidence; it supplies no F/S lifecycle comparison and does not falsify abstract persistent-state reuse or the homogeneous-volume model.
  -> # Architecture — FNTO Dynamic Transport Reuse

**Current as of:** 2026-11-13. **Role:** current physical/representation contracts, candidate status and unresolved design decisions. See [document ownership](README.md).

## 1. Objective and design boundary

Maintain physically credible global il ...[cut]...  current reference semantics, explicit ballistic readout, near-contact spatial requirements and geometry-state invalidation are later requirements, not retroactive contents or validation of v0.1. A materially different detailed learned implementation needs an explicitly versioned successor design if adopted; the present candidate plan is not that adoption.

No architecture decision or scientific verdict changes through document reorganisation. Superseded design detail and the pre-cleanup source versions remain in the [historical archive](history/living_document_cleanup_2026-10-29/README.md).
RUN [Read baseline roles strategy doc] cd /c/Projects/ReST-GI && cat docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md | cut -c1-2500
RUN [Read Architecture doc] cd /c/Projects/ReST-GI && cat docs/Architecture.md | cut -c1-3000
RUN [Find roadmap entries for WL86-96A] grep -n -i -E "WL8[6-9]|WL9[0-9]|ECRO|86A|CK2|M64|resolvent|learned" docs/RESEARCH_ROADMAP.md | cut -c1-400 | head -120
  -> 1:# Research Roadmap — Dynamic Learned Transport Reuse
7:**Retained local-scattering decision (human, 2026-11-10):** The learned ER closure qualified through WL82–84 is the **primary development baseline for local material scattering**, not the selected final ReST-GI architecture. Its demonstrated s ...[cut]... d.
164:- Consider interpretable dependency discovery and optional learned scoring only after physical coverage and representation feasibility are established, with algorithm-only and oracle controls.
165:- Make quality-matched external comparisons, including TransGI, only after reference, workload and accounting alignment. Plan durations from measured costs; venue planning does not relax gates. WL67 supplied aligned physical/reference comparisons, but a path-traced re-render is **not** a Neural Full Recompute. Any learned dynamic-update superiority claim requires an actually invalidated extern
RUN [Find roadmap entries for WL86-96A] cd /c/Projects/ReST-GI && grep -n -i -E "WL8[6-9]|WL9[0-9]|ECRO|86A|CK2|M64|resolvent|learned" docs/RESEARCH_ROADMAP.md | cut -c1-400 | head -120
RUN [Read roadmap ECRO section in full] sed -n '126,160p' docs/RESEARCH_ROADMAP.md
  -> <persisted-output>
Output too large (34.5KB). Full output saved to: C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bo98jlwoa.txt

Preview (first 2KB):
## Current decision boundary

**Current decision (human, 2026-11-13):** ECRO is reinstated as  ...[cut]... s, independent references, matched explicit comparator and prospective gates before an authorised evidence run. Do not tune WL86 retrospectively, promote an explicit candidate as the project goal, or launch an automatic successor batch.

**GM1 evidence record (measured; nothing adopted; decisions remain human).**

- [WL85](worklogs/85_WORKLOG_GM1_MECHANISM_NEUTRAL_TRANSPORT_CHARACTERIZATION.md) (2026-11-11): mechanism-neutral characterisation on an open fixture. Its frozen workload qualification is UNSUITABLE by one clause, so its conclusions are formally INDETERMINATE.
...
</persisted-output>
RUN [Read roadmap ECRO section in full] cd /c/Projects/ReST-GI && sed -n '126,160p' docs/RESEARCH_ROADMAP.md
READ C:\Users\dna10\.claude\projects\c--Projects-ReST-GI\46d65e15-e083-4057-875f-9282fb68aa5e\tool-results\bo98jlwoa.txt 
RUN [List headings of WL86, 86A, 87] for f in docs/worklogs/86_* docs/worklogs/86A_* docs/worklogs/87_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-160; done
  -> === docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
1:# Worklog — ECRO v0: ER-conditioned Learned Global Transport Response (2026-11-11)
3:## Origin and scope
13:## ECRO v0 as implemented (frozen in `experiments/ecro86/protocol/ecro86.json`)
45:## Workload and gates (prospective)
58:## Pre-freeze d ...[cut]... n `protocol/gm1_explicit87.json`)
69:## Prospective rules (summary; full text in the protocol)
102:## Pre-freeze development (non-evidence)
134:## Evidence run (freeze `60efecb`; LabServer63; 2026-11-11 08:38–10:15 UTC)
150:## Final report (Korean)
152:### A. CURRENT ASSESSMENT
164:### B. IMPLEMENTATION FACT
189:### C. MEASUREMENT
292:### D. OBSERVATION
304:### E. INTERPRETATION
329:### F. UNRESOLVED QUESTIONS
338:### G. ARCHITECTURE VERDICT (scope: one room family, rigid box edits, three frozen ER materials, point / cone sources, 512², E1 / E2 historical, E3 untouch
356:### H. REPRODUCIBILITY
RUN [List headings of WL86, 86A, 87] cd /c/Projects/ReST-GI && for f in docs/worklogs/86_* docs/worklogs/86A_* docs/worklogs/87_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-160; done
RUN [Read WL86 implementation, gates and verdict sections] sed -n '13,57p;123,133p;232,280p' docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md | cut -c1-1800
  -> ## ECRO v0 as implemented (frozen in `experiments/ecro86/protocol/ecro86.json`)

**Operator contract.**

- T = S_ER C_G, q = S_ER e. The reference integrator truncates at D = 6 scattering events: b = Σ_{j=0}^{5} T^j q.
- LOW = q + T q is computed physically (camera vertex x1 with next-event estimati ...[cut]... oor in E2's C |
| GM1 ARCHITECTURE VIABILITY | **NOT SUPPORTED under the frozen gates** | Cause: learned-operator error on sharp / moderate receivers of the held-out edit. Not a representation failure and not an instability |

ECRO v0 is not adopted. No successor or rescue was started. The decision about the frozen region rule, the sharp-receiver error and any further batch belongs to the human owner.

### Living documents

- `docs/Architecture.md` is not changed.
- `docs/RESEARCH_ROADMAP.md` receives one dated evidence record under the current decision boundary (measured result, no adoption).
RUN [Read WL86 implementation, gates and verdict sections] cd /c/Projects/ReST-GI && sed -n '13,57p;123,133p;232,280p' docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md | cut -c1-1800
RUN [Read WL87 assessment, interpretation, verdict] sed -n '152,163p;292,356p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-2200
  -> ### A. CURRENT ASSESSMENT

- **기존 불확실성.**
  - WL86의 ECRO는 LOW + table lookup으로 path tracing 대비 per-frame 9배의 이득을 보였습니다. 그러나 이것이 학습 덕분인지, table 표현 덕분인지 분리된 적이 없습니다.
  - WL86A는 sharp 실패의 원인으로 두 가지를 제시했습니다. 하나는 학습된 single flight의 결손이고, 다른 하나는 512-ray state relation의 sampling 한계입니다. 둘은 분리되지 않았습니다.
  - W ...[cut]... 성립합니다. 학습 없는 동일-state 계산이 학습 operator와 같거나 나은 fidelity를, 비슷한 edit 비용과 같은 frame 비용으로 냅니다. H1은 source-only 갱신에서만 성립합니다. H2는 출력 표현이 아니라 수신측 512-ray relation의 sharp-family 표본 배치에 한정해 부분적으로 성립하며, 학습과 explicit 모두에 공통입니다. 학습 operator 연구를 지지하는 측정은 source-only 갱신 비용 하나이고, 이번 결과는 explicit physical baseline을 이 workload의 기준점으로 세웁니다. 이후 아키텍처 선택은 하지 않습니다 |

**Completion outcome.**
- 1 (EXPLICIT SUFFICIENCY SUPPORTED, 이 workload와 0.10–0.03 및 g 수준 한정)에 해당합니다.
- 여기에 4의 요소가 함께 있습니다. 학습 효용은 source-only 갱신에서만 측정되었고, sharp 표현성은 state 표본에 의존해 미결입니다.
- ECRO v0의 WL86 verdict와 WL86A 귀속은 그대로입니다.

### H. REPRODUCIBILITY
RUN [Read WL87 assessment, interpretation, verdict] cd /c/Projects/ReST-GI && sed -n '152,163p;292,356p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-2200
RUN [Read WL86A final report] sed -n '191,250p' docs/worklogs/86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md | cut -c1-1800
  -> ## Final report

### Implementation fact

- Diagnostics only, on the frozen WL86 operators: exact recomputation; importance decomposition of the rendered error over sender patches and bins; seven controlled incident fields per configuration; identical-encoding beam pairs; frozen-operator evaluations ...[cut]... ict unchanged) | cause now localised; a diagnostic result does not upgrade it |

**Completion outcome.** The cause is identified at mechanism level and is not a single isolated variable: outcome 4 (cause not reducible to one factor) for the learning side, with an element of outcome 3 (the frozen state relation, not the source encoding, is inadequate for sharp-sender flights). No recovery was attempted. The next architecture decision is the human owner's.

### Living documents

- `docs/Architecture.md` is not changed.
- `docs/RESEARCH_ROADMAP.md`: one sentence added to the WL86 evidence record.
RUN [Read WL86A final report] cd /c/Projects/ReST-GI && sed -n '191,250p' docs/worklogs/86A_WORKLOG_ECRO_SHARP_FAILURE_ATTRIBUTION.md | cut -c1-1800
RUN [List headings of WL92-93D worklogs] for f in docs/worklogs/92_* docs/worklogs/92A_* docs/worklogs/92B_* docs/worklogs/93_* docs/worklogs/93A_* docs/worklogs/93B_* docs/worklogs/93C_* docs/worklogs/93D_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-200; done
  -> === docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md
1:# Worklog — ECRO Sharp Interreflection: Representability, Identifiability and Learning-Failure Attribution (2026-11-13)
3:## Origin and scope
14:## ECRO v0 as recovered from the implementation
37:## Design of t ...[cut]... an)
=== docs/worklogs/93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md
1:# Worklog — ECRO Frozen Local Operator Ownership Attribution (WL93D) (2026-11-14)
3:## Origin and scope
22:## Protocol (`experiments/ecro_local93d/protocol/ecro_local93d.json`)
104:## Dev chain (before the freeze; not evidence) and a disclosed pre-freeze exposure
129:## Freeze and execution
146:## Measurements
208:## Observation
223:## Interpretation
234:## Unresolved questions
242:## Visuals (`results/evaluation/93D/`, manifest with hashes)
250:## Reproducibility
266:## Final report (Korean, for the owner)
RUN [List headings of WL92-93D worklogs] cd /c/Projects/ReST-GI && for f in docs/worklogs/92_* docs/worklogs/92A_* docs/worklogs/92B_* docs/worklogs/93_* docs/worklogs/93A_* docs/worklogs/93B_* docs/worklogs/93C_* docs/worklogs/93D_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-200; done
RUN [Read WL92 recovered operator, gates, verdict] sed -n '14,36p;187,198p;242,286p' docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md | cut -c1-1800
  -> ## ECRO v0 as recovered from the implementation

Source: `experiments/ecro86/model86.py`, `train86.py`, `gen86.py`, `pt86.py` (read, not reconstructed from worklogs).

**Operator.**
- `u = c·s`, `Bq = a·u`, `E1` = flight of `Bq` through the 512 state rays with weight cos dΩ / κ, `B = a·(c·E1) + a·(P ...[cut]... 
- 기존 ECRO 연산자 계열은 배포 interface의 정보만으로 sharp 방향 상호반사를 규칙 수준까지 재현할 정보 용량을 가집니다(B SUPPORTED).
- 표현 용량은 사전 게이트로 확립되지 않았습니다. 재질 단위 rank-16은 부족하고, 기하별 factor는 고정 기하에서 충분해 보이지만 게이트 fit이 수렴 규칙을 충족하지 못했습니다.
- 따라서 물리적으로 식별 가능한 감독이 일반화를 개선하는지는 **이 batch에서 검정하지 않았습니다**.
- 사전 completion 코드는 1(G_rep 실패)입니다. 이는 "표현 한계가 입증됨"이 아니라 "표현 적합성이 확립되지 않음"으로 읽어야 합니다(프로토콜 문구 결함, 공개).
- 다음 결정이 정당하게 바꿀 수 있는 항목: 사전 규칙으로 확립된 것은 없습니다(**none without further evidence**). 서술적 증거는 학습 / 감독 계약과 generator의 기하 조건부 국소 응답 쪽을 가리키며, state 표본 한계가 부차 요인입니다. 연산자 표현 계열 교체를 요구하는 증거는 없습니다.
- ECRO v0의 WL86 판정은 그대로입니다. 어떤 아키텍처 전환도 승인되지 않았습니다.
RUN [Read WL92 recovered operator, gates, verdict] cd /c/Projects/ReST-GI && sed -n '14,36p;187,198p;242,286p' docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md | cut -c1-1800
RUN [Read WL92A and WL92B verdict sections] sed -n '213,226p;267,308p' docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md | cut -c1-1700; echo ----------92B; sed -n '133,169p;203,244p' docs/worklogs/92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md | cut -c1-1700
  -> ### Verdict (frozen mapping)

| Criterion | Result |
|---|---|
| OWNERSHIP_IMPROVED | **PASS** (ratios 0.42–0.54) |
| PANEL_LOCAL_IMPROVED | **PASS** (retention 0.21 vs 0.03 / 0.08) |
| HELDOUT_SHARP | **not passed (E4 not admitted)**; numerically OWN 0.013 / 0.012 vs SUM 0.025 / 0.024 |
| NO_REGRES ...[cut]... . 증거가 지지하는 개입 | 학습된 **source 측 국소 응답의 기하 조건부 생성**과 **tied-factor 소유(같은 `a cᵀ`의 이중 사용)**, 그리고 별도로 **학습 분포 밖 source에서의 global 항**. source interface 변경은 지지되지 않음. 구체적 설계는 책임자 결정이며, 구현하지 않았습니다 | 상호작용 증거 |

**Completion outcome: C (multiple interacting limitations established; E5 qualified).**
- WL92A의 D_STOPPED 판정은 그대로이고, E4는 계속 미승인입니다.
- OWN은 component 소유 개선으로는 유효하지만, 독립 workload에서 sharp fidelity의 일관된 우위는 확립되지 않았습니다.
- 다음 아키텍처 결정이 다룰 수 있는 항목: 국소 각 응답(source 측) × tied-factor 소유 × 기하 조건부 generator 동작의 **상호작용**, 그리고 미학습 source class에서의 global 항. source interface는 해당하지 않습니다.
- 어떤 개입도 채택하거나 시작하지 않았습니다.
RUN [Read WL92A and WL92B verdict sections] cd /c/Projects/ReST-GI && sed -n '213,226p;267,308p' docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md | cut -c1-1700; echo ----------92B; sed -n '133,169p;203,244p' docs/worklogs/92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md | cut -c1-1700
RUN [List results directory and WL92B results] ls results/ | tr '\n' ' ' ; echo; ls -R results/ecro_cf92b | head -60
  -> 8dna_deformation_replication 8dna_dynamic_baseline 8dna_replication CHECKPOINTS_SHA256.txt arch_study86b audit97 baseline_sup67 batch1_hq_dynamic_failure ecro86 ecro86a ecro_attr92 ecro_attr95a ecro_cf92b ecro_exp94b ecro_local93d ecro_local93d_checker ecro_mat94 ecro_mat94r ecro_own92a ecro_q94c ec ...[cut]... cf_E1_T1_LSPOT.txt
cf_E2_T0.txt
cf_E2_T1.txt
cf_E4_T0.txt
cf_E4_T1.txt
e5_E5k995_T0.txt
e5_E5k995_T1.txt
e5_E5k995_T1_LMIX.txt
e5_E5k995_T1_LSPOT.txt

results/ecro_cf92b/results/logs:
A_final.log
A_final.rc
S0_repro.log
S0_repro.rc
S1_e5gen.log
S1_e5gen.rc
S1_ofield_LSPOT.log
S1_ofield_LSPOT.rc
S1_split_LSPOT.log
S1_split_LSPOT.rc
S2_cf_E1_T1_LSPOT.log
S2_cf_E1_T1_LSPOT.rc
S2_cf_E2_T0.log
S2_cf_E2_T0.rc
S2_cf_E2_T1.log
S2_cf_E2_T1.rc
S2_cf_E4_T0.log
S2_cf_E4_T0.rc
S2_cf_E4_T1.log
S2_cf_E4_T1.rc
S2_local.log
S2_local.rc
S2_tests.log
S2_tests.rc
S3_render_E1_T1_LSPOT.log
S3_render_E1_T1_LSPOT.rc
RUN [List results directory and WL92B results] cd /c/Projects/ReST-GI && ls results/ | tr '\n' ' ' ; echo; ls -R results/ecro_cf92b | head -60
RUN [Inspect WL92B final json structure] python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
def walk(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if depth<2: print('  '*depth+str(k), type(v).__name__, (len(v) if hasattr(v,'__len__') else v))
            walk(v,p+'/'+str(k),depth+1)
walk(d)
EOF
  -> repro dict 5
  checkpoints dict 6
  tables dict 24
  renders dict 42
  field dict 6
  PASS bool True
tests dict 6
  T_CF_IDENTITY dict 3
  T_LINEAR dict 3
  T_EVENT dict 5
  T_DISJOINT dict 2
  T_REPRO dict 1
  ALL_PASS bool True
cases dict 5
  E2_T1 dict 6
  E2_T0 dict 6
  E1_T1_LSPOT dict 6
  E4_T1 dict 6
  E4_T0 dict 6
local dict 4
  metrics dict 30
  geometry_variation dict 6
  seed_variation dict 3
  config_variation dict 0
hypotheses dict 8
  H_local dict 2
  H_global dict 2
  single_term dict 2
  local_source_side dict 2
  local_receiver_side dict 2
  H_source dict 4
  H_geometry dict 3
  H_interact dict 1
e5 dict 6
  candidates dict 3
  admissions list 1
  chosen str 3
  models dict 6
  controls dict 4
  criteria dict 3
outcome dict 4
  label str 1
  supported list 1
  baseline_reproduced bool True
  e5_admitted bool True
RUN [Inspect WL92B final json structure] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
def walk(o,p='',depth=0):
    if isinstance(o,dict):
        for k,v in o.items():
            if depth<2: print('  '*depth+str(k), type(v).__name__, (len(v) if hasattr(v,'__len__') else v))
            walk(v,p+'/'+str(k),depth+1)
walk(d)
EOF
RUN [Print E5 model results from raw json] python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
e=d['e5']
print(json.dumps(e['models'],indent=0)[:3000])
print(json.dumps(e['controls'],indent=0)[:1500])
print(e['criteria']); print(d['outcome'])
EOF
  -> {
"ecro_s8601": {
"fidelity": {
"T0": {
"stationary_all": {
"E_H": 0.004783305789655724,
"signed": -0.00371690085096192,
"H_share": 0.22509041124446755,
"noise_bound": 0.0026565018963836053,
"pixels": 224909,
"capture": 0.021250597762961315
},
"A": {
"E_H": 0.01386652421693996,
"signed ...[cut]... onary_all', 0.02605741534408915, 0.01561449225237798]], 'PASS': False}, 'DYNAMIC': {'F3_lost': [], 'C_edit': {'ecro_s8601': [0.9956623971655678, 0.9723835664837693, 'PASS'], 'ecro_s8602': [0.8449629428551005, 0.9637465986992403, 'PASS'], 'sum_s8601': [0.7642477837214108, 0.9677120122421736, 'PASS'], 'sum_s8602': [0.8861656971397109, 0.9760736394706454, 'PASS'], 'own_s8601': [0.8806789503306752, 0.9697864834661369, 'PASS'], 'own_s8602': [0.7531823139674269, 0.9794100472530711, 'PASS']}, 'PASS': True}}
{'label': 'C', 'supported': ['H_geometry'], 'baseline_reproduced': True, 'e5_admitted': True}
RUN [Print E5 model results from raw json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
e=d['e5']
print(json.dumps(e['models'],indent=0)[:3000])
print(json.dumps(e['controls'],indent=0)[:1500])
print(e['criteria']); print(d['outcome'])
EOF
RUN [Print E5 controls and hypotheses from raw json] python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
e=d['e5']
for k,v in e['controls'].items():
    print(k, {r:round(v[r]['E_H'],4) for r in v if isinstance(v[r],dict) and 'E_H' in v[r]})
for m,v in e['models'].items():
    print(m, 'T0 C',round(v['fidelity']['T0']['C']['E_H'],4),'T1 C',round(v['fidelity']['T1']['C']['E_H'],4), 'panel T1', round(v['fidelity']['T1']['obj_panel']['E_H'],4))
print(json.dumps(d['hypotheses'],indent=0)[:2500])
print(e['admissio
  -> ORACLE_T0 {'stationary_all': 0.0024, 'A': 0.0075, 'C': 0.0099, 'D': 0.0025, 'obj_panel': 0.0124, 'obj_sphere': 0.0084}
ORACLE_T1 {'stationary_all': 0.0024, 'A': 0.0047, 'C': 0.0062, 'D': 0.0025, 'obj_panel': 0.0082, 'obj_sphere': 0.0083}
X_EXP_T0 {'stationary_all': 0.0039, 'A': 0.0143, 'C': 0.0174 ...[cut]... ct_change': {'value': 0.017927519969122757, 'PASS': True}, 'Q3_C_sharp_moderate': {'value': 0.3066032181126305, 'sm_share': 0.8304013906646923, 'pixels': 4980, 'PASS': True}, 'Q4_D_broad_multibounce': {'value': 0.025987707868927564, 'absH_over_abs': 0.5100945200265075, 'absH': 0.013256187371989697, 'absH_floor': 0.0010133121290362848, 'PASS': True}, 'Q6_H_share': {'value': 0.230490999598903, 'PASS': True}, 'Q7_uncertainty': {'value': 0.0023718485722185293, 'PASS': True}, 'SHARP_FAMILY': {'share': 0.09812653942617543, 'half_split_noise': 0.0009287932882892508, 'PASS': True}}, 'ADMITTED': True}]
RUN [Print E5 controls and hypotheses from raw json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/ecro_cf92b/results/final92b.json'))
e=d['e5']
for k,v in e['controls'].items():
    print(k, {r:round(v[r]['E_H'],4) for r in v if isinstance(v[r],dict) and 'E_H' in v[r]})
for m,v in e['models'].items():
    print(m, 'T0 C',round(v['fidelity']['T0']['C']['E_H'],4),'T1 C',round(v['fidelity']['T1']['C']['E_H'],4), 'panel T1', round(v['fidelity']['T1']['obj_panel']['E_H'],4))
print(json.dumps(d['hypotheses'],indent=0)
RUN [Read WL93D observation, interpretation, protocol] sed -n '208,241p' docs/worklogs/93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md | cut -c1-2000; echo; sed -n '22,60p' docs/worklogs/93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md | cut -c1-900
  -> ## Observation

- **Geometry dependence is real but small.**
  - The learned S1 operator changes between geometry pairs by 4–8 % of the physical response overall, and by 9–14 % on the E2 / E5 box-move pairs. That is consistent with WL92B's 14.5 % in a different norm.
  - The change is 3–6 × the Lamb ...[cut]... e use: s16. Receiver use: flight(q_B5).
  - The bases are identical for every geometry.
- Metrics:
  - E_phys: error against M64.
  - D_geo: drift over geometry pairs. The pairs are E2, E5k995 and E6 T0↔T1, plus validation derangement pairs (k, k+1).
  - D_seed: seed-to-seed difference.
  - S_geo: the geometry-varying share of the squared physical error.
  - Error by incident ring.

**Task C — incident-subspace attribution.**
- On evaluation incident fields, the error is split into in-span and out-of-span parts relative to the fixed material-level basis.
- Geometry drift is split the same way.
RUN [Read WL93D observation, interpretation, protocol] cd /c/Projects/ReST-GI && sed -n '208,241p' docs/worklogs/93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md | cut -c1-2000; echo; sed -n '22,60p' docs/worklogs/93D_WORKLOG_ECRO_FROZEN_LOCAL_OPERATOR_OWNERSHIP_ATTRIBUTION.md | cut -c1-900
RUN [Read WL93D addendum (CK2 scene definition)] sed -n '1,11p;26,60p;127,162p' docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md | cut -c1-1500
  -> # Worklog — WL93D Post-hoc Qualitative Addendum: Checkerboard Reflected by the Sharp Panel (2026-11-14, revision 2)

> **POST-HOC QUALITATIVE ADDENDUM to WL93D (non-numbered).** Not part of the frozen WL93D protocol, evidence or labels, and nothing here changes them. WL93D's labels stand: REPRESENTA ...[cut]...  one of 69 state entries through `mat_mlp`, and per-material training (each training patch's material was fixed per object). Neither is tested.
- How much of the arms' remaining rendered under-response (Table 2) beyond the table-level compression comes from bin-specific B values toward the panel, rather than bin sums? This is not separated.
- The WL93A T0 family was not rendered.

## Architecture Verdict

None. No label is computed. WL93D's frozen labels and evidence are unchanged. Nothing is adopted and no successor is authorised. The addendum's status (closed or not) is the owner's decision.
RUN [Read WL93D addendum (CK2 scene definition)] cd /c/Projects/ReST-GI && sed -n '1,11p;26,60p;127,162p' docs/worklogs/93D_WORKLOG_ADDENDUM_CHECKERBOARD_QUALITATIVE.md | cut -c1-1500
RUN [List headings WL94-96A] for f in docs/worklogs/94_* docs/worklogs/94A_* docs/worklogs/94B_* docs/worklogs/94C_* docs/worklogs/94D_* docs/worklogs/95_* docs/worklogs/95A_* docs/worklogs/96_* docs/worklogs/96A_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-200; done
  -> === docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md
1:# Worklog — ECRO Material Generalization vs. Sharp Angular Representation (WL94) (2026-11-14)
3:## Origin and scope
25:## Implementation Fact (verified in code before the design)
65:## Design (frozen)
122:## Measurement
 ...[cut]... 96A: Learned Higher-Order Transport Utility Ceiling (2026-11-16)
3:## Origin and scope
17:## Event families (verified; Stage 0)
35:## A. Implementation fact
63:## B. Measurement
65:### B1. Tail size and reference noise
71:### B2. Integrated ROI (paired vs F; `s1.json`)
94:### B3. Field level, by family and region (`tails.json`; table = W_T-weighted patch × bin × channel rel L1; n0 = reference half-split)
122:### B4. Lifecycle cost (Stage 3; `cost_s1.0.json`)
166:## C. Observation
184:## D. Interpretation
207:## E. Architecture verdict
216:## F. Reproducibility
254:## Disclosures and deviations
RUN [List headings WL94-96A] cd /c/Projects/ReST-GI && for f in docs/worklogs/94_* docs/worklogs/94A_* docs/worklogs/94B_* docs/worklogs/94C_* docs/worklogs/94D_* docs/worklogs/95_* docs/worklogs/95A_* docs/worklogs/96_* docs/worklogs/96A_*; do echo "=== $f"; grep -n "^#" "$f" | cut -c1-200; done
RUN [Read WL94 and WL94A interpretation and verdicts] sed -n '236,294p' docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | cut -c1-1500; echo ====94A; sed -n '124,173p' docs/worklogs/94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md | cut -c1-1200
  -> ## Observation

1. **Material response compression is not sharp-specific.**
   - The frozen operator compresses the albedo response at Lambert box patches in every component (S and G), on every face and in every bin group.
   - The compression appears equally in the sharp reader (ROI_T) and in the d ...[cut]... kept relative to A on s8602, but worse than frozen own on both.
  - Textured sharp ROI: worse than A on s8601, mixed on s8602; contrast improved.
  - Panel-face global term: collapsed on s8601 (−0.42), kept on s8602 (−0.07).
  - Forgetting as the main cause: not established (seed-dependent).
  - Sharp-specific residual: the panel-sender delivery to the panel-lit face plus the panel-mediated global term, independent of material generalization.
- No adoption, no full training and no successor. WL94's H-M / H-A labels are unchanged. The decision returns to the owner and the supervising assistant.
RUN [Read WL94 and WL94A interpretation and verdicts] cd /c/Projects/ReST-GI && sed -n '236,294p' docs/worklogs/94_WORKLOG_ECRO_MATERIAL_GENERALIZATION_VS_SHARP_ANGULAR.md | cut -c1-1500; echo ====94A; sed -n '124,173p' docs/worklogs/94A_WORKLOG_ECRO_MATERIAL_FIDELITY_RETENTION_SMOKE.md | cut -c1-1200
RUN [Read WL94D verdict and WL95 failure attribution] sed -n '143,184p' docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md | cut -c1-1500; echo ====95; sed -n '169,240p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1600
  -> ## Observation

- Direct source-specific supervision does move the learned S1 sender toward the teacher on the fields it is trained on: teacher loss −25 % / −11 %, field error −11 %, mirror-peak capture 0.23 → 0.32. The move is mostly an **energy increase** (validation energy ratio 0.87 → 0.99 / 0.9 ...[cut]...    - The "exact sender" is M64(s). It carries L1, L2 and L12 as well as L4, and is read through the deployed fixed 512-ray state (L5, L6).
   - The decomposition is **OPEN** (§2.3).
6. **The CK2 panel-face correlation −0.334** [FACT] is a Pearson correlation over 9 patches for one light. The ideal-mirror footprint covers 3 of those patches [INF].
7. **Minor** [MEAS]:
   - T_SEM s8601 is 1.51e-7 in `audit.json` but printed as 7.6e-8 in 94D:42. Both are far below the 1e-5 threshold.
   - WL92's protocol wording "rows sum to the directional albedo" omits a z_bi/dΩ factor (`diag92.py:172`) [FACT].
RUN [Read WL94D verdict and WL95 failure attribution] cd /c/Projects/ReST-GI && sed -n '143,184p' docs/worklogs/94D_WORKLOG_ECRO_SHARP_SENDER_ORACLE_SUPERVISED_SMOKE.md | cut -c1-1500; echo ====95; sed -n '169,240p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1600
RUN [Read WL95 attribution by stage] sed -n '241,331p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1300
  -> ### 2.3 Attribution by stage

**Local sender, learned**
- [MEAS] At the zero-operator level relative to M64 (§2.1). The learned panel Bq vs the exact physical q: rel L1 0.90–1.14, flux 0.16–1.07 (WL92:131; WL92B CF_E2_T1).
- [MEAS] S1 S_geo is 0.2–0.8 % (WL93D, `final93d.json`). The learned S1 opera ...[cut]...  cannot be assumed to solve the problem.
- **Sharp lobe and energy**
  - [MEAS] S1 maximum directional albedo is 0.825 with the true weight.
  - [MEAS] L1-optimal rank-16 fits under-emit (energy about 0.7) and Frobenius fits are about 1.09. Peak and L1 must therefore be read together with energy (§2.1).
- **Propagation error accounting**
  - [FACT] E_H is an absolute-value metric. Step differences between arms are not additive error shares (WL91 caveat).
  - [FACT] Field rel-L1, operator norms and E_H are distinct norms (auditB §1c). Every attribution in §6 is therefore reported per norm.

---
RUN [Read WL95 attribution by stage] cd /c/Projects/ReST-GI && sed -n '241,331p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1300
RUN [Read WL95 recommended hypothesis, verdict, canon] sed -n '582,603p;779,804p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1800; sed -n '151,168p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1200
  -> ## 5. Recommended Hypothesis

**None.** Recommending one now would mean selecting on theoretical expressiveness. Six reasons:

1. [INF] **G1 is inadmissible as the method under current canon**, by close analogy with Architecture.md:13 and roadmap:128 (§1.5). Its formal status is owner Gate A (§7). U ...[cut]...  project goal".
- [FACT] The owner question "is a learning-free global solve around ER a candidate or a baseline?" (`docs/plans/gm1_architecture_review/60_coordinator_synthesis.md:94-100`) is constrained by the 2026-11-13 decision, which leans towards "comparator". It is not formally closed.
- [FACT] Baseline-role terminology (AGENTS.md; `docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md:23-32`) defines four separate roles:
  1. Neural Full Recompute;
  2. Native Reuse;
  3. same-checkpoint F/S;
  4. path-traced re-render.

  Beating a path tracer is not superiority over Neural Full Recompute.

---
RUN [Read WL95 recommended hypothesis, verdict, canon] cd /c/Projects/ReST-GI && sed -n '582,603p;779,804p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1800; sed -n '151,168p' docs/worklogs/95_WORKLOG_SHARP_TRANSPORT_REPRESENTATION_ARCHITECTURE_DESIGN.md | cut -c1-1200
NOTE: Now WL95A in full-ish and WL96.
RUN [Read WL95A facts, measurements, verdict] sed -n '17,48p;53,100p;116,189p' docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-1500
  -> ## A. Implementation Fact

**Verified contracts (Stage 0, all PASS; `results/ecro_attr95a/s0.json`).**

| test | result |
|---|---|
| T_BINS | 4 rings × 16 sectors; `pt86.Ctx.bin_of` = iz·16 + iφ on 4,096 random directions (agreement 1.0). Corrected W_T = ((d//16) + ½)/4 · 2π/64 sums to π. `an94.W`  ...[cut]... nce.

Nothing is adopted. No successor is started. The decision returns to the owner.

## Unresolved questions

- Whether any fixed-basis local carrier (more bins, adaptive bins, a lobe basis) removes the outgoing bias on CK2 without the exact estimator's variance. Untested; not authorised.
- Whether a grazing-inclusive training distribution changes the rank-16 conclusion. The fit distribution has 4.3 % ring-0 irradiance.
- The ROI-level ranking of sender vs higher-order errors at higher render spp.
- Generality: other scenes, lights, area lights, sharp movers and non-rigid edits are untested.
NOTE: Now WL95A in full-ish and WL96.
RUN [Read WL95A facts, measurements, verdict] cd /c/Projects/ReST-GI && sed -n '17,48p;53,100p;116,189p' docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md | cut -c1-1500
RUN [Read WL96A origin, event families, implementation] sed -n '1,62p' docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1700
  -> # Worklog — WL96A: Learned Higher-Order Transport Utility Ceiling (2026-11-16)

## Origin and scope

- **Brief.** "WL96A — Learned Higher-Order Transport Utility Ceiling" (user). A suffix of WL96: the dependent, learning-free feasibility test of the one learned element WL96 left standing (a broad ≥  ...[cut]... ns and e3 parts only k ≥ 3.
  - **T_REF_RECON.** The reference families sum to the WL95A floor table (2.1e-8).
  - **T_EXP_DEPLOYED.** The explicit state equals `build_state` bitwise.
  - **T_EXP_SPLIT.** T_p + T_o = T and T(L2 + L3 + L4) = L3 + L4 + L5 to 2.2e-7; linearity holds to 2.2e-7; outputs are non-negative.
  - **T_READOUT.** masks94 is identical to WL95A's, and a re-render of the WL95A floor table equals the WL95A file to a maximum per-pixel 1.7e-7 of the image mean.
- **Existing behaviour preserved.** No production source was edited. The diagnostics live only in the WL96A namespace.
RUN [Read WL96A origin, event families, implementation] cd /c/Projects/ReST-GI && sed -n '1,62p' docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1700
RUN [Read WL96A measurements, interpretation, verdict] sed -n '63,215p;254,290p' docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1700
  -> ## B. Measurement

### B1. Tail size and reference noise

- **Field.** At box A, the reference e3 exitance (e3_p 25.7 + e3_o 133.3) is as large as e2 (151.3); the tail is about 51 % of box A's table.
- **ROI image.** Dropping the tail (arm Z) changes the ROI by 0.178 of the ROI total (E_H 0.175). Th ...[cut]... ; only the render spp was reduced. The figure therefore showed qualitatively that |XOD − F| and |XOP − F| were the smallest tail differences, and that G and Z were larger. No number was read.
  - All criteria were written before any WL96A code ran and were not changed.
- **Post-dev fixes before the freeze:** a module import order, numpy-bool JSON serialisation, and pool sizes.
- **No amendment.**
- **Post-hoc, figure only:** figW3 colours (`figw3_posthoc.py`).
- **The full-run `T_READOUT` record** stores the masks check under a key containing "sha". The pass flag includes it (masks identical).
RUN [Read WL96A measurements, interpretation, verdict] cd /c/Projects/ReST-GI && sed -n '63,215p;254,290p' docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md | cut -c1-1700
RUN [Read WL96 origin, verdict, evidence reconciliation] sed -n '3,75p' docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-1900
  -> ## Origin, scope and process

- **Brief.** "WL96 — Spatio-Angular Transport Carrier Architecture Design" (user).
  - It is a regular numbered, design-only batch.
  - Nothing was implemented or trained, and no renderer or GPU job was run.
  - No production code, checkpoint or canonical document was c ...[cut]... ransfer rank (≥ 256) and d3's local rank (~500 at 1,024 bins) all bound a *linear, light-independent, separable, fixed-basis* form, in Frobenius norms that down-weight grazing rows.
  - [MEAS-S, physics critic] Re-weighting rows to unit outgoing flux halves d1's CK2 requirement: rank 128 → 0.118.
  - d1's "a non-negative fit cannot be better in that norm" is false. In WL95 §2.1, L1-NMF16 scores 0.380 against signed svd16 0.891.
  - These calculations falsify ECRO's u = c·s, Bq = a·u form at fine resolution. They do **not** prove that no compact learned representation can carry sharp structure.
RUN [Read WL96 origin, verdict, evidence reconciliation] cd /c/Projects/ReST-GI && sed -n '3,75p' docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-1900
RUN [Read WL96 candidates, critique, judgment] sed -n '92,252p' docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-1700
  -> ## B. Independent candidate designs (complete contracts in the plan files)

**Two kinds of thing.** Eight named candidates reduce to two kinds (both critics agree):
- **K1.** Learned *sharp* carriers, which put the sharp first order inside a learned operator.
- **K2.** Explicit, learning-free sharp  ...[cut]... plit by sender class and by delta vs extended incidence (same ER response everywhere) compatible with `Architecture.md:21`?
- **Smoke.** Whether to commission the §F smoke, and whether its WL89 W1 arm requires a new split.
- **Sharp learned C.** Whether to commission a sharp learned-C test at all. It is untested, with a negative WL52 prior, and was not proposed as a candidate here.

## Proposed living-document amendment

None. No architecture is adopted or superseded.

A dated roadmap record of this batch's closure is added, per the AGENTS.md batch-closing rule. `Architecture.md` is unchanged.
RUN [Read WL96 candidates, critique, judgment] cd /c/Projects/ReST-GI && sed -n '92,252p' docs/worklogs/96_WORKLOG_SPATIO_ANGULAR_TRANSPORT_CARRIER_DESIGN.md | cut -c1-1700
RUN [List WL96 plan files and physics critic headings] ls docs/plans/wl96_spatio_angular_carrier/ && wc -c docs/plans/wl96_spatio_angular_carrier/critic_*.md && grep -n "^#" docs/plans/wl96_spatio_angular_carrier/critic_physics.md | cut -c1-200
  -> README.md
critic_cost.md
critic_physics.md
designer1_free.md
designer2_lobe_state.md
designer3_basis.md
designer4_continuous.md
26880 docs/plans/wl96_spatio_angular_carrier/critic_cost.md
24804 docs/plans/wl96_spatio_angular_carrier/critic_physics.md
51684 total
1:# WL96 adversarial critique: physic ...[cut]... . ER ports and deterministic oracles
74:## 2. Rank and capacity claims
76:### 2.1 What each claim measures
90:### 2.2 Re-check of the weighting [MEAS-C]
107:### 2.3 Consequence for WL95A's RANK16 verdict
113:## 3. The "learned outgoing field = baked radiance" argument (d4 §0.1)
124:## 4. Event-family ownership and double counting
145:## 5. Falsifiability and supervision
159:## 6. Mechanisms the designers missed or dismissed too quickly
177:## 7. Per-candidate verdicts
248:## 8. Recommended WL96 judgment: **EVIDENCE_INSUFFICIENT**
262:## 9. The single bounded smoke that most reduces uncertainty
RUN [List WL96 plan files and physics critic headings] cd /c/Projects/ReST-GI && ls docs/plans/wl96_spatio_angular_carrier/ && wc -c docs/plans/wl96_spatio_angular_carrier/critic_*.md && grep -n "^#" docs/plans/wl96_spatio_angular_carrier/critic_physics.md | cut -c1-200
RUN [Read physics critic bottom line, rank claims, judgment] sed -n '7,33p;74,123p;159,176p;248,290p' docs/plans/wl96_spatio_angular_carrier/critic_physics.md | cut -c1-1600
  -> ## 0. Bottom line

1. **The four designers converge on one structure. They are not four architectures.** [INF]
   - An explicit, learning-free sharp first order: node carrier + photon relay (d1), DQ + relays (d2), light-side photons or a fine table (d3), CIS + DQ (d4). These are WL95 G1-class estima ...[cut]...  GPU-utilisation probe.
   - Add an in-sample non-negative rank-24 fit of the explicit tail, labelled a feasibility probe, not a bound.

**Frozen mapping**
- H_L ≤ tolerance and sweeps ≤ generator time → every Λ variant is NO_UTILITY, i.e. NO_VIABLE_LEARNED_MECHANISM on the tested class.
- H_L > tolerance → one Λ hypothesis proceeds to supervised-target construction (d4's tally), after Gate A.
- Stage 2 unresolvable → the sharp-carrier choice has no image-level target on CK2.

**Outside this smoke** (separable later if wanted): the photon/DQ estimator comparison, α ladders and sharp learned C.
RUN [Read physics critic bottom line, rank claims, judgment] cd /c/Projects/ReST-GI && sed -n '7,33p;74,123p;159,176p;248,290p' docs/plans/wl96_spatio_angular_carrier/critic_physics.md | cut -c1-1600
RUN [Scan cost critic for judgment and scope statements] grep -n "^#" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-200; grep -n -i -E "judg|verdict|NO_VIABLE|EVIDENCE_INSUFF|Neural Full|baseline role|scal" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-420 | head -50
  -> 1:# WL96 adversarial critique: cost, scaling, dynamic reuse and the explicit comparator
7:## 0. Bottom line
32:## 1. Cross-cutting cost findings (number re-checks)
34:### 1.1 Arithmetic the designers got right [MEAS-C re-computation]
46:### 1.2 Numbers that are wrong, inflated or hidden
64:### 1.3 D ...[cut]... ation of the ceiling on an exact prefix is missing.
270:- **Comparisons.** Internal only. Neural Full Recompute is UNMEASURED.
294:  - Same GPU (LabServer63), wall time and GPU-kernel time, with a full-scale utilisation probe per the standing rule.
295:  - Measured stages: sweeps, Λ apply, and ECRO `factors`, at N = 1,270 / 2,802 / 5,080 (WL91 scaling geometry).
299:- **Confirms NO_VIABLE_LEARNED_MECHANISM** when all of the following hold:
305:- **COST_ONLY candidate** (a scaling hypothesis needing a training smoke at large N) when U_max ≥ 1.25 at the largest N while H_acc is within the bound.
RUN [Scan cost critic for judgment and scope statements] cd /c/Projects/ReST-GI && grep -n "^#" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-200; grep -n -i -E "judg|verdict|NO_VIABLE|EVIDENCE_INSUFF|Neural Full|baseline role|scal" docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-420 | head -50
RUN [Read cost critic bottom line, utility, judgment, smoke] sed -n '7,31p;102,142p;252,310p' docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-1500
  -> ## 0. Bottom line

1. **There are eight named candidates but only two kinds of thing.**
   - Every "learned" candidate that survives (d1 SIC-R Λ_θ, d2 SRR-LB Λ_θ, d3 LSC-R and ESC-R R_θ, d4 LIR Λ_θ) has the same learned component: a non-negative, low-rank, geometry-conditioned resolvent for ≥ 3-even ...[cut]... next step is owner Gate A.
- **ONE_FALSIFIABLE_CANDIDATE (learned resolvent; training needs separate authorisation)** when H_acc exceeds the bound in a predeclared sharp-reader region on both workloads.
- **COST_ONLY candidate** (a scaling hypothesis needing a training smoke at large N) when U_max ≥ 1.25 at the largest N while H_acc is within the bound.
- **STOP** if the tallies do not sum to the reference (≤ 1e-6), or the explicit sweeps fail to reproduce WL87 / WL89 values.

**Budget [INF].**
- No new PT renders if the W1 split exists; otherwise 2 × 4,096 spp for W1.
- NMF fits: CPU minutes.
RUN [Read cost critic bottom line, utility, judgment, smoke] cd /c/Projects/ReST-GI && sed -n '7,31p;102,142p;252,310p' docs/plans/wl96_spatio_angular_carrier/critic_cost.md | cut -c1-1500
RUN [Print WL96A protocol JSON] python - <<'EOF'
import json
d=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
def show(o,ind=0,maxlen=900):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)):
                print(' '*ind+str(k)+':'); show(v,ind+2)
            else: print(' '*ind+f'{k}: {str(v)[:maxlen]}')
    elif isinstance(o,list):
        for v in o:
            if isinstance(v,(dict,list)): show(v,ind+2)
            else: print(' '*ind+'- '+str(v)[:max
  -> protocol: ecro_util96a
version: 1
title: WL96A - learned higher-order transport utility ceiling (learning-free; CK2 none)
brief: user brief 'WL96A - Learned Higher-Order Transport Utility Ceiling' (2026-11-16)
batch_class: suffix of WL96 (dependent evidence-gap follow-up of its NO_VIABLE_LEARNED ...[cut]...  R_pool 2, 2 / 4 render reps at 32 spp, 32-spp extension halves, 8 Stage 2 lights at 32 spp, 200 fit steps, 8-spp lifecycle frames; every path incl. extension, Stage 2 and scaling forced); outcome lines were suppressed ('S1 DEV', 'FINAL DEV'); Stage 0 test values were read. DISCLOSED: the dev figure figW1 was opened to check layout; because the dev tails use the REAL CKF_none reference and the real explicit operator (only the render spp was reduced), it showed qualitatively that |arm - F| in the ROI is visibly smallest for XOD / XOP among the tail arms and larger for the learned G arms and for
RUN [Print WL96A protocol JSON] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('experiments/ecro_util96a/protocol/ecro_util96a.json'))
def show(o,ind=0,maxlen=900):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)):
                print(' '*ind+str(k)+':'); show(v,ind+2)
            else: print(' '*ind+f'{k}: {str(v)[:maxlen]}')
    elif isinstance(o,list):
        for v in o:
            if isinstance(v,(dict,list)): show(v,ind+2)
            else: prin
RUN [List WL96A and WL95A result files] ls results/ecro_util96a results/ecro_attr95a | head -60 && python - <<'EOF'
import json,os
p='results/ecro_util96a'
for f in os.listdir(p):
    if f.endswith('.json'):
        d=json.load(open(os.path.join(p,f)))
        print(f, list(d.keys())[:30] if isinstance(d,dict) else type(d))
EOF
  -> results/ecro_attr95a:
an.json
chain.status
deliv.npz
gpu.txt
main.json
ops95a.npz
s0.json

results/ecro_util96a:
T96A.npz
chain.status
cost_s1.0.json
final.json
gpu.txt
s0.json
s1.json
tails.json
tails.npz
cost_s1.0.json ['s', 'N', 'gpu', 'foreign_before', 'frame_spp', 'res', 'arms', 'components', 'rays', 'process_gpu_mem', 'U', 'boundary_ok']
final.json ['stage0', 'S1', 'S1_ext', 'S2', 'accuracy_confirmed', 'cost_met', 'U', 'LEARNED_RESOLVENT_UTILITY']
s0.json ['T_PMAT', 'T_KMAX', 'T_SPLIT_FIELD', 'T_SPLIT_TAG', 'T_REF_RECON', 'T_EXP_DEPLOYED', 'T_EXP_SPLIT', 'T_READOUT', 'ALL']
s1.json ['images', 'decision', 'ext']
tails.json ['ext', 'R_pool', 'deployed_rays_to_panel', 'state_rays', 'field', 'magnitude', 'timing_s', 'total_s', 'gpu_mem_peak_mib_torch']
RUN [List WL96A and WL95A result files] cd /c/Projects/ReST-GI && ls results/ecro_util96a results/ecro_attr95a | head -60 && python - <<'EOF'
import json,os
p='results/ecro_util96a'
for f in os.listdir(p):
    if f.endswith('.json'):
        d=json.load(open(os.path.join(p,f)))
        print(f, list(d.keys())[:30] if isinstance(d,dict) else type(d))
EOF
RUN [Verify WL96A reported numbers against raw json] python - <<'EOF'
import json
p='results/ecro_util96a/'
f=json.load(open(p+'final.json')); print({k:f[k] for k in ['S1','accuracy_confirmed','cost_met','U','LEARNED_RESOLVENT_UTILITY']})
s=json.load(open(p+'s1.json'))
print(json.dumps(s['decision'],indent=0))
im=s['images']
print(type(im), list(im.keys())[:20] if isinstance(im,dict) else '')
for k,v in im.items():
    if isinstance(v,dict):
        print(k, {a:(round(b,5) if isinstance(b,float) else b) for a,b in v.items() if not isinstance(b,(di
  -> {'S1': 'S1_NO_HEADROOM', 'accuracy_confirmed': False, 'cost_met': False, 'U': {'1.0': {'LAM24': {'edit': 0.9321302015681585, 'source': 1.040492233086946}, 'LAM64': {'edit': 0.9209615109936242, 'source': 1.038301742024448}, 'ZERO': {'edit': 1.038028573853174, 'source': 1.0428847966464583}}}, 'LEARNED ...[cut]...  0.003310844302177429}, "parts_sum_over_wall": 0.999701665478876, "n": 16}, "dirty_rays_median": 49936.0, "torch_peak_mib": 250.57373046875}, "LAM24": {"edit": {"wall_s": 0.15462632570415735, "wall_raw_s": 0.15462632570415735, "compile_s_total": 0.0, "events_with_compilation": 0, "cpu_s": 0.5801017204999983, "gpu_kernel_ms": 25.504400418605655, "parts": {"generator": 0.008362113498151302, "geometry_update": 0.03027352597564459, "lambda_apply": 0.00020901300013065338, "prefix_order": 0.001994163729250431, "relation_coefficients": 0.00506
[] True NVIDIA GeForce RTX 3080 Ti, 595.71.05, 12288 MiB
RUN [Verify WL96A reported numbers against raw json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
p='results/ecro_util96a/'
f=json.load(open(p+'final.json')); print({k:f[k] for k in ['S1','accuracy_confirmed','cost_met','U','LEARNED_RESOLVENT_UTILITY']})
s=json.load(open(p+'s1.json'))
print(json.dumps(s['decision'],indent=0))
im=s['images']
print(type(im), list(im.keys())[:20] if isinstance(im,dict) else '')
for k,v in im.items():
    if isinstance(v,dict):
        print(k, {a:(round(b,5) if isinstance(b,float) else b) for a,b in v.items
RUN [Inspect WL95A raw analysis json] python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(list(a.keys())[:40])
def find(o,path='',depth=0,out=[]):
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),depth+1,out)
    elif isinstance(o,(int,float)):
        out.append((path,o))
    return out
out=find(a)
import re
for p,v in out:
    if re.search(r'(A4|X1|q256|A0_s860[12]|M64).*(boxA|box_A)',p) and re.search('pool|deploy',p) and re.search('ratio|rel',p):
        print(p, rou
  -> ['delivery', 'flight', 'rank16', 'images', 'floor_E_H', 'components', 'verdicts', 'D_used']
/delivery/pool|A0_s8601|boxA/ratio 0.5173
/delivery/pool|A0_s8601|boxA/abs_rel_l1 0.5692
/delivery/pool|A0_s8601|boxA/peak_amp_ratio 0.5926
/delivery/pool|A0_s8601|boxA/value_at_ref_peak_ratio 0.3026
/de ...[cut]... livery/pool|q256|boxA/peak_amp_ratio 1.5947
/delivery/pool|q256|boxA/value_at_ref_peak_ratio 1.0589
/delivery/pool|q256|boxA/per_draw/ratio/mean 1.4221
/delivery/pool|q256|boxA/per_draw/ratio/median 1.3817
/delivery/pool|q256|boxA/per_draw/ratio/p90 1.7713
/delivery/pool|q256|boxA/per_draw/ratio/worst 1.2403
/delivery/pool|q256|boxA/per_draw/abs_rel_l1/mean 0.7099
/delivery/pool|q256|boxA/per_draw/abs_rel_l1/median 0.6571
/delivery/pool|q256|boxA/per_draw/abs_rel_l1/p90 1.0739
/delivery/pool|q256|boxA/per_draw/abs_rel_l1/worst 1.2955
/delivery/pool|q256|boxA/se_abs_rel_l1 0.0335
684
RUN [Inspect WL95A raw analysis json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
a=json.load(open('results/ecro_attr95a/an.json'))
print(list(a.keys())[:40])
def find(o,path='',depth=0,out=[]):
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),depth+1,out)
    elif isinstance(o,(int,float)):
        out.append((path,o))
    return out
out=find(a)
import re
for p,v in out:
    if re.search(r'(A4|X1|q256|A0_s860[12]|M64).*(boxA|box_A)',p) and re.search('pool|deploy',p) and re.search('ratio|rel
RUN [Inspect WL95A delivery arrays and flight attribution] python - <<'EOF'
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype)
a=json.load(open('results/ecro_attr95a/an.json'))
print(a['D_used'], json.dumps(a['verdicts'])[:1500])
print(json.dumps(a['flight'])[:2500])
EOF
  -> dep|A1 (1270,) float64
dep|A2 (1270,) float64
dep|A3 (1270,) float64
dep|A4 (1270,) float64
dep|A0_s8601 (1270,) float64
dep|A0_s8602 (1270,) float64
dep|M64_spool (1270,) float64
dep|q256 (1270,) float64
dep|X1c (1270,) float64
dep|X1 (1270,) float64
draws|A1 (64, 1270) float64
draws|A2  ...[cut]...  0.46446359806088566, "Rs": 0.024175660096948215, "Rs_se": 0.034055179336480924, "Ra": 0.23188168674623544, "Ra_se": 0.046620248374830725, "Rp": 0.1571067426298993, "Rp_se": 0.06009806987301328, "Rq_incident": 0.06643679185456292, "Rq_incident_se": 0.0053191013018510916, "Rq_outgoing": 0.16544489489167252, "Rq_outgoing_se": 0.046560463778183264, "E_X1_pool": 0.05129950858780274, "E_M64_spool_pool": 0.44028793796393745, "E_X1c_pool": 0.20840625121770204, "E_q256_pool": 0.37385114610937453, "E_A4_pool": 0.4871664042591579, "verdict": "ANGULAR_PROJECTION_EVIDENCED", "near_boundary_extend": false}
RUN [Inspect WL95A delivery arrays and flight attribution] cd /c/Projects/ReST-GI && python - <<'EOF'
import numpy as np, json
z=np.load('results/ecro_attr95a/deliv.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype)
a=json.load(open('results/ecro_attr95a/an.json'))
print(a['D_used'], json.dumps(a['verdicts'])[:1500])
print(json.dumps(a['flight'])[:2500])
EOF
RUN [Check where deployed draw falls in pooled draw distribution (WL95A)] python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz')
box=slice(1130,1184)
ref=0.5*(z['ref_a']+z['ref_b'])[box]
for arm in ['A4','q256','X1','X1c','M64_spool','A1','A0_s8601']:
    dr=z['draws|'+arm][:,box]; dep=z['dep|'+arm][box]
    r=dr.sum(1)/ref.sum()
    rd=dep.sum()/ref.sum()
    print(arm,'dep ratio %.3f'%rd,'draw ratio: mean %.3f sd %.3f min %.3f p10 %.3f med %.3f p90 %.3f max %.3f'%(r.mean(),r.std(ddof=1),r.min(),np.percentile(r,10),np.median(r),np.percentile(
  -> A4 dep ratio 1.208 draw ratio: mean 1.544 sd 0.283 min 1.078 p10 1.185 med 1.514 p90 1.905 max 2.422 frac draws below dep 0.141 SE mean 0.035
    pooled L1 0.546; dep L1 0.524
q256 dep ratio 1.077 draw ratio: mean 1.422 sd 0.270 min 0.954 p10 1.064 med 1.382 p90 1.771 max 2.240 frac draws below de ...[cut]... .141 SE mean 0.063
    pooled L1 0.268; dep L1 0.388
M64_spool dep ratio 1.158 draw ratio: mean 1.495 sd 0.276 min 1.043 p10 1.141 med 1.459 p90 1.856 max 2.363 frac draws below dep 0.125 SE mean 0.035
    pooled L1 0.500; dep L1 0.483
A1 dep ratio 0.921 draw ratio: mean 0.975 sd 0.076 min 0.807 p10 0.866 med 0.971 p90 1.062 max 1.179 frac draws below dep 0.234 SE mean 0.010
    pooled L1 0.655; dep L1 0.623
A0_s8601 dep ratio 0.495 draw ratio: mean 0.517 sd 0.036 min 0.442 p10 0.472 med 0.520 p90 0.569 max 0.592 frac draws below dep 0.328 SE mean 0.005
    pooled L1 0.569; dep L1 0.588
RUN [Check where deployed draw falls in pooled draw distribution (WL95A)] cd /c/Projects/ReST-GI && python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/deliv.npz')
box=slice(1130,1184)
ref=0.5*(z['ref_a']+z['ref_b'])[box]
for arm in ['A4','q256','X1','X1c','M64_spool','A1','A0_s8601']:
    dr=z['draws|'+arm][:,box]; dep=z['dep|'+arm][box]
    r=dr.sum(1)/ref.sum()
    rd=dep.sum()/ref.sum()
    print(arm,'dep ratio %.3f'%rd,'draw ratio: mean %.3f sd %.3f min %.3f p10 %.3f med %.3f p90 %.3f max %.3f'%(r.mean(),r.std(ddof=1),r.min(),np.percentile(r,10),n
RUN [Check WL91 lifecycle anchors and WL87 win/loss counts] sed -n '226,248p' docs/worklogs/91_WORKLOG_GM1_C3URX1_HELDOUT_VALIDATION_AND_ECONOMICS.md | cut -c1-700; grep -n -E "우세|열세|0 / 3|36" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-500 | head -20
  -> | | C3URX1 | C3UR | R0 | Path tracing |
|---|---|---|---|---|
| Edit to frame, ms | 124.7 | 138.2 | 145.4 | 250.1 |
| Edit without render, ms | 69.4 | 83.0 | 89.8 | 30.9 |
| Source update to frame, ms | 75.8 | 92.2 | 85.4 | 213.1 |
| Source update without render, ms | 21.4 | 37.4 | 31.0 | 0.0 |
| Tw ...[cut]... o} f(v, ω_r)dv · cos ω_r · ΔΩ/8 을 씁니다(65,536 MIS 샘플).
208:| D | 0.0033 / 0.0049 / 0.0036 / 0.0028 | 0.0041 / 0.0043 / 0.0034 / 0.0024 | 0.0050 / 0.0059 / 0.0052 / 0.0043 |
216:- 학습 대 explicit. 36개 region-class 비교(3 edit × T0/T1 × 6 class) 결과입니다.
266:- **per-edit 고정비용(wall).** PT 36 ms, ECRO 69–78 ms, EXP 89–95 ms, TF64 152 ms, TF256 334 ms, TF1024 1.2 s, TF4096 4.6 s, C3 0.43–0.58 s.
346:| LEARNED OPERATOR ADDITIONAL UTILITY | **SUPPORTED (사전 규칙) — 근거는 source만 바뀌는 갱신 비용 하나** | fidelity 효용 없음(우세 0 / 열세 3 / 36). edit-to-first-frame ≤ 15 % 차이(기준 미달). source-only table 갱신 3.5–3.9× 저렴(약 40–50 ms) |
RUN [Check WL91 lifecycle anchors and WL87 win/loss counts] cd /c/Projects/ReST-GI && sed -n '226,248p' docs/worklogs/91_WORKLOG_GM1_C3URX1_HELDOUT_VALIDATION_AND_ECONOMICS.md | cut -c1-700; grep -n -E "우세|열세|0 / 3|36" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-500 | head -20
RUN [Read WL87 measurement tables for learned vs explicit and cost] sed -n '196,232p;262,290p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
  -> - zero / no-visibility 0, 선형성 1.5e-7, furnace 6e-7.
- null edit, move-and-return, selective = full 은 B가 비트 동일입니다.
- RR 대 PT 기대값 0.02 % 이내.
- 보고용(이산화 진단): 512-ray quadrature의 directional albedo 오차 median은 panel S1 5.3 %(max 18 %), B 1.0 %, S2 2.0 % 입니다.

**충실도 (E_H, T1. EXP / ECRO s8601 / s8602 / ora ...[cut]... time-to-quality.**
  - E2 C 0.05: PT 2,440 spp, EXP 5,485 spp(bias floor 0.049 근처), ECRO 미도달(floor 0.072), C3 224, TF4096 190.
  - E2 C 0.03: PT 7,412 spp, C3 1,470, TF1024 1,525, TF4096 683. EXP / ECRO / TF256은 미도달입니다.
  - D 영역에서는 모든 table arm이 PT의 약 0.6× spp로 같습니다.
- **amortisation (V frames, 0.03).**
  - V = 1: PT 1.4–2.5 s, ECRO 0.27–0.48 s, EXP 0.29–0.49 s.
  - V = 16: PT 21–40 s, ECRO 3.3–6.5 s, EXP 3.3–6.3 s.
- **cold.** PT 첫 렌더 21 s(컴파일), table 0.7–0.9 s. W tables 2.6 s(첫 프로세스).
- **memory.** state 17.6 MB, W 3.9 MB, ECRO factors 42.9 MB(735,664 params), table 1 MB, process 3.9–4.9 GB.
RUN [Read WL87 measurement tables for learned vs explicit and cost] cd /c/Projects/ReST-GI && sed -n '196,232p;262,290p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
RUN [Inspect WL96 ring allocation calc] ls results/wl96_calc results/wl95_calc | head -40; python - <<'EOF'
import json
d=json.load(open('results/wl96_calc/ring_alloc.json'))
print(json.dumps(d)[:2500])
EOF
  -> results/wl95_calc:
S1_ops.npz
wl95_calc.json
wl95_calc.py
wl95_calc2.json
wl95_calc2.py
wl95_calc3.json
wl95_calc3.py

results/wl96_calc:
ck2_support.py
ck2_support_E.npy
ring_alloc.json
ring_alloc.py
scratch
{"rank_MP_ring0_c0": 16, "rank_MP_ring0_c1": 16, "rank_MP_ring0_c2": 16, "ring0_nonneg_rank ...[cut]... p_l1": 0.3172547735878435, "ring3_op_l1": 0.21634569377636717, "ck2_support_field_rel_l1": 0.8919760730459726, "ck2_support_energy": 0.11135769060676808, "ck2_support_peak": 0.0020368713184459202}, "A3": {"ring0_op_l1": 0.5181423088584282, "ring1_op_l1": 0.3824184549524946, "ring2_op_l1": 0.16098370412968546, "ring3_op_l1": 0.18689875916459792, "ck2_support_field_rel_l1": 0.3633457849015464, "ck2_support_energy": 0.668681490523468, "ck2_support_peak": 0.07643962083456857}, "row_weight_share_by_ring_in_flux_norm": [0.06359803703826138, 0.1699202996348839, 0.31379206706219476, 0.45268959626466]}
RUN [Inspect WL96 ring allocation calc] cd /c/Projects/ReST-GI && ls results/wl96_calc results/wl95_calc | head -40; python - <<'EOF'
import json
d=json.load(open('results/wl96_calc/ring_alloc.json'))
print(json.dumps(d)[:2500])
EOF
RUN [List WL86 results] ls results/ecro86 results/ecro86/results 2>/dev/null | head -50
  -> results/ecro86:
chain.status
chain.status.run1
chain_frozen_run1.status
code
results
watch.out
watch_run1.out

results/ecro86/results:
amend1
contracts
econ
final.json
logs
masks
models
qualify.json
qualify_frozen.json
renders
stage1.json
tests
viz
RUN [List WL86 results] cd /c/Projects/ReST-GI && ls results/ecro86 results/ecro86/results 2>/dev/null | head -50
RUN [Verify WL86 E2 region C numbers from raw final.json] python - <<'EOF'
import json
d=json.load(open('results/ecro86/results/final.json'))
print(list(d.keys()))
def find(o,path='',out=None):
    if out is None: out=[]
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),out)
    elif isinstance(o,list):
        for i,v in enumerate(o): find(v,path+'/'+str(i),out)
    else: out.append((path,o))
    return out
out=find(d)
print(len(out))
import re
for p,v in out:
    if re.search(r'E2',p) and re.search(r'/C/',p) and re.searc
  -> ['seeds', 'passing_seeds', 'STATUS', 'economics_seed']
2281
/seeds/ecro_s8601/fidelity/E2/T0/C/E_H 0.060280104370449034
/seeds/ecro_s8601/fidelity/E2/T0/C/noise_bound 0.01013178292799979
/seeds/ecro_s8601/fidelity/E2/T1/C/E_H 0.07171602311030288
/seeds/ecro_s8601/fidelity/E2/T1/C/noise_bound 0. ...[cut]... 02/stale/E2/E_H/C/E_H 0.08082335342266597
/seeds/ecro_s8602/stale/E2/E_H/C/H_share 0.7094348708180728
/seeds/ecro_s8602/stale/E2/E_H/C/noise_bound 0.01255009966510753
/seeds/ecro_s8602/stale/E2/E_H/C/hl_over_h 0.8956383028933693
/seeds/ecro_s8602/stale/E2/E_H/C/capture 0.11392638950699716
/seeds/ecro_s8602/ablation/E2/nog/C/E_H 0.45305739741345935
/seeds/ecro_s8602/ablation/E2/nog/C/noise_bound 0.010762633933862967
/seeds/ecro_s8602/ablation/E2/nos/C/E_H 0.3171177166830949
/seeds/ecro_s8602/ablation/E2/nos/C/noise_bound 0.011264564929181124
/STATUS STOPPED_GENERALIZATION_NOT_SUPPORTED
RUN [Verify WL86 E2 region C numbers from raw final.json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json
d=json.load(open('results/ecro86/results/final.json'))
print(list(d.keys()))
def find(o,path='',out=None):
    if out is None: out=[]
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),out)
    elif isinstance(o,list):
        for i,v in enumerate(o): find(v,path+'/'+str(i),out)
    else: out.append((path,o))
    return out
out=find(d)
print(len(out))
import re
for p,v in out:
    if re.search(r'E2',p) and re.sea
RUN [Find WL87 realization distribution evidence] ls results/gm1_explicit87 results/gm1_explicit87/results 2>/dev/null | head -40; grep -n -E "0\.024|realization|75 ?%" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-400 | head -20
  -> results/gm1_explicit87:
chain.status
code
gpu_util_trace.txt
results

results/gm1_explicit87/results:
e3_admission.json
e3_choice.json
econ
final87.json
lists
logs
masks
renders
stage1_87.json
stage2_87.json
stage3_87.json
tables
tests
viz
143:| 16 realizations × 6 configurations | 0.9 min | 24 % |  ...[cut]... 출력 표현은 sharp family를 담을 수 있습니다(WL86 R1과 일치).
333:- pooled state가 남기는 E2 C의 체계적 결손(−0.024)의 원인은 분리하지 않았습니다. 후보는 수신 patch의 공간 평균, outgoing bin 평균, 512-방향 공유 offset 규칙입니다.
343:| FROZEN STATE ADEQUACY | **CONDITIONALLY SUPPORTED** | deployed state로 모든 gate 통과(ACCURATE). 그러나 E2에서 realization 통과율 75 % < 90 %, broad는 통과 |
344:| SHARP TRANSPORT REPRESENTABILITY | **INDETERMINATE** | E2 C에서 pass/fail이 realization마다 다르고(0.024–0.077), deployed와 pooled는 통과. 체계적 잔여 결손(pooled −0.024)과 표본 분산이 함께 있음. sender-side 배치는 같은 예산으로 해소 |
362:  - Quadrature 87001 (+1000 per realization), dense 87002, q 87003, C3 87004.
RUN [Find WL87 realization distribution evidence] cd /c/Projects/ReST-GI && ls results/gm1_explicit87 results/gm1_explicit87/results 2>/dev/null | head -40; grep -n -E "0\.024|realization|75 ?%" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-400 | head -20
RUN [Read WL87 state realization table] sed -n '243,258p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
  -> **State realization (Stage 2; 16개 독립 realization, 같은 규칙과 예산).**

| | E2 T1 C E_H (mean, SD, range) | signed mean [95 % CI] | deployed | pooled 8,192 ray | C3 |
|---|---|---|---|---|---|
| EXP | 0.044, 0.018, 0.024–0.077 | −0.024 [−0.042, −0.006] | −0.049 | −0.024 (E_H 0.030) | −0.005 |
| ECRO (같은 st ...[cut]...  | −0.070 | — | — |

- E2 C에서 EXP의 realization 통과율(≤ 0.06)은 75 % 입니다. E1과 E3에서는 100 % 입니다.
- deployed state는 분포의 나쁜 꼬리에 있습니다: signed −0.049, realization 평균 −0.024.
- **family (light → panel → Lambert).**
  - 중요 relation 중 하나도 hit하지 못한 비율은 56–62 %(가중 38–65 %)이며, 모든 realization에서 비슷합니다.
  - 중요 patch당 panel hit는 min 0, median 13–20 입니다.
  - BIN read / truth: deployed 0.73–0.97, realization 평균 0.88–1.07(SD 0.13–0.30).
  - patch-level relative RMS: BIN 0.41–0.69, RAY 0.39–0.97.
- 분류(사전 규칙, region C).
  - E2 T1: D_MIXTURE (A systematic + C receiver sampling). pooled가 평균 오차의 100 %를 유지하고 CI는 0을 제외합니다.
RUN [Read WL87 state realization table] cd /c/Projects/ReST-GI && sed -n '243,258p' docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md | cut -c1-900
RUN [Find passivity measurements across worklogs] grep -n -i -E "passiv|albedo [0-9]|directional albedo" docs/worklogs/86_*.md docs/worklogs/86A_*.md docs/worklogs/92_*.md docs/worklogs/92A_*.md docs/worklogs/92B_*.md docs/worklogs/93*.md docs/worklogs/94*.md docs/worklogs/95A_*.md | cut -c1-360 | head -40
  -> docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md:41:- Zero input → zero, exact superposition and non-negative output hold by construction. No single-scattering passivity is imposed on the multi-flight response.
docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md:90:-  ...[cut]... t bins (directional albedo up to 2.3 / 3.1). The bound operators and M64 are passive (≤ 0.96).
docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:143:- **Learned sender (A).** On CK2 it is a large error source in delivery: 0.52× / 1.63× flux and a misplaced footprint. It is seed-signed, as in WL94C/94D, and non-passive. It is far from M64 at the field level, and slightly worse than the rank-16 bounds in rel L1 but not 
docs/worklogs/95A_WORKLOG_SHARP_SENDER_REPRESENTABILITY_AND_FLIGHT_ATTRIBUTION.md:161:  | arm | passive | flux vs A4 | map vs A4 | ROI E_H vs A4 |
RUN [Find passivity measurements across worklogs] cd /c/Projects/ReST-GI && grep -n -i -E "passiv|albedo [0-9]|directional albedo" docs/worklogs/86_*.md docs/worklogs/86A_*.md docs/worklogs/92_*.md docs/worklogs/92A_*.md docs/worklogs/92B_*.md docs/worklogs/93*.md docs/worklogs/94*.md docs/worklogs/95A_*.md | cut -c1-360 | head -40
RUN [Find albedo computation in WL95A code] grep -n -i "albedo\|passive" experiments/ecro_attr95a/a95.py | head -30
  -> 148:    """operator [NB_in, NB_out, 3] vs M64: flux-L1 (zb_in zb_out), mass, directional albedo max (W_T), min entry."""
151:            'mass': float((A * ZZ[..., None]).sum() / (M * ZZ[..., None]).sum()), 'albedo_max': float(alb.max()),
152:            'albedo_mean': float(alb.mean()), 'min_entry' ...[cut]... onneg': bool(min(p['min_entry'] for p in per) >= 0),
500:        c1 = mj['operator'][X]['passive']
504:        rk[X] = {'passive': c1, 'flux_ratio_vs_A4': pX['ratio'] / p4['ratio'], 'map_rel_l1_vs_A4': mapdiff, 'E_H': img[f'pool_{X}']['E_H'],
511:            return ((not r['passive']) or abs(r['flux_ratio_vs_A4'] - 1) > 0.1 + 2 * 0.02 or r['map_rel_l1_vs_A4'] > 0.10 + 0.02
559:    ax[3].bar(ok_, [om[k]['albedo_max'] for k in ok_]); ax[3].axhline(1, color='r', ls='--', label='passivity limit')
560:    ax[3].set_title('operator max directional albedo (W_T)', fontsize=8); ax[3].legend(fontsize=7)
RUN [Find albedo computation in WL95A code] cd /c/Projects/ReST-GI && grep -n -i "albedo\|passive" experiments/ecro_attr95a/a95.py | head -30
RUN [Read albedo metric code and operator metrics] sed -n '140,153p;300,320p' experiments/ecro_attr95a/a95.py | cut -c1-260; python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(list(m.keys()))
op=m.get('operator',{})
for k,v in op.items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items() if not isinstance(b,(list,dict))})
EOF
  -> """S1 field [42, NB, 3]: flux rel L1 / energy / peak of X against reference Y."""
    Wd = W[None, :, None]
    pk = Y.argmax(1)[:, None, :]
    return {'rel_l1': float((np.abs(X - Y) * Wd).sum() / (Y * Wd).sum()), 'energy': float((X * Wd).sum() / (Y * Wd).sum()),
            'peak': float(np.take_a ...[cut]... do_max': 0.8057, 'albedo_mean': 0.3286, 'min_entry': 0.0, 'passive': True, 'nonneg': True}
A3 {'op_flux_l1': 0.2331, 'mass': 0.8005, 'albedo_max': 0.8456, 'albedo_mean': 0.4558, 'min_entry': 0.0, 'passive': True, 'nonneg': True}
A4 {'op_flux_l1': 0.0, 'mass': 1.0, 'albedo_max': 0.8247, 'albedo_mean': 0.6349, 'min_entry': 0.0009, 'passive': True, 'nonneg': True}
A0_s8601 {'op_flux_l1_mean': 1.0179, 'albedo_max': 2.2637, 'passive_all': False, 'nonneg': True, 'mass_mean': 0.869}
A0_s8602 {'op_flux_l1_mean': 0.987, 'albedo_max': 3.0774, 'passive_all': False, 'nonneg': True, 'mass_mean': 0.835}
RUN [Read albedo metric code and operator metrics] cd /c/Projects/ReST-GI && sed -n '140,153p;300,320p' experiments/ecro_attr95a/a95.py | cut -c1-260; python - <<'EOF'
import json
m=json.load(open('results/ecro_attr95a/main.json'))
print(list(m.keys()))
op=m.get('operator',{})
for k,v in op.items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items() if not isinstance(b,(list,dict))})
EOF
RUN [Inspect stored operators in WL95A ops file] python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/ops95a.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype)
EOF
grep -n "ops95a\|WT =\|^WT\|ZZ =\|zb =\|def wt\|W94 =" experiments/ecro_attr95a/a95.py | head -20
  -> A1 (64, 64, 3) float64
A2 (64, 64, 3) float64
A3 (64, 64, 3) float64
A4 (64, 64, 3) float64
4:  python a95.py stage0 <R> <R92A>     contract tests, WL94D old-convention reproduction, bound operators (results/s0.json, ops95a.npz)
27:WT = ZB * DOM                                         # corrected cos dOmega (4 rings x 16 sectors)
28:W94 = ((np.arange(NB) // 8) + 0.5) / 8 * DOM          # an94.W as coded (8 x 8)
29:ZZ = ZB[:, None] * ZB[None, :]
256:    np.savez_compressed(os.path.join(R, 'results', 'ops95a.npz'), A1=ops['nmf16_wl92'], A2=ops['nmf16_l1irls'], A3=A3, A4=Mq[1])
258:    jd({'status': st_, 'tests': T, 'ops_sha256': MT.sha_file(os.path.join(R, 'results', 'ops95a.npz')),
270:    ops = dict(np.load(os.path.join(R, 'results', 'ops95a.npz')))
RUN [Inspect stored operators in WL95A ops file] cd /c/Projects/ReST-GI && python - <<'EOF'
import numpy as np
z=np.load('results/ecro_attr95a/ops95a.npz',allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype)
EOF
grep -n "ops95a\|WT =\|^WT\|ZZ =\|zb =\|def wt\|W94 =" experiments/ecro_attr95a/a95.py | head -20
RUN [Check the 9-40x claim and section-F deviation disclosures in WL96A] grep -n -E "9.{0,3}40|40 ?×|40x" docs/worklogs/96A_*.md docs/RESEARCH_ROADMAP.md | cut -c1-300; grep -n -i "deviat\|section F\|§F\|§ F" docs/worklogs/96A_*.md | cut -c1-300
  -> docs/worklogs/96A_WORKLOG_LEARNED_HIGHER_ORDER_UTILITY_CEILING.md:178:- **The existing learned global term** (descriptive; different input and ownership) is 9–40× further from the floor than explicit e3_o (ROI dI 0.025 / 0.013, under-delivering, signed −0.019 / −0.009). It also has field erro
docs/R ...[cut]... GFUL_HEADROOM**. The higher-order tail is large
5:- **Brief.** "WL96A — Learned Higher-Order Transport Utility Ceiling" (user). A suffix of WL96: the dependent, learning-free feasibility test of the one learned element WL96 left standing (a broad ≥ 3-event resolvent behind a physically correct sharp prefix). It closely follows WL96 §F, restr
117:- **Field-level secondary rule** (WL96 §F bound max(2 SE, n0 / 2), reported only):
213:- The sharp-prefix relevance part of WL96 §F (oracle vs binned e2_p at higher spp) was **not** part of this brief and remains open.
254:## Disclosures and deviations
RUN [Check the 9-40x claim and section-F deviation disclosures in WL96A] cd /c/Projects/ReST-GI && grep -n -E "9.{0,3}40|40 ?×|40x" docs/worklogs/96A_*.md docs/RESEARCH_ROADMAP.md | cut -c1-300; grep -n -i "deviat\|section F\|§F\|§ F" docs/worklogs/96A_*.md | cut -c1-300
RUN [Check baseline-terminology gate usage in WL86-96A protocols] grep -l -i "neural full recompute" experiments/*/protocol/*.json 2>/dev/null | head -30; echo ---; grep -o -i ".\{120\}neural full recompute.\{160\}" experiments/gm1_explicit87/protocol/*.json experiments/ecro86/protocol/ecro86.json experiments/ecro_util96a/protocol/*.json 2>/dev/null | head -8; echo ---; grep -n -i "neural full recompute\|UNMEASURED\|external" docs/worklogs/86_*.md docs/worklogs/87_*.md docs/worklogs/96A_*.md docs/worklogs/95A_*.md docs/worklogs/92*.md | cut -c1-300 | head -20
  -> experiments/ecro86/protocol/ecro86.json
experiments/gm1_c3urx1_91/protocol/gm1_c3urx1_91.json
---
---
docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md:11:- **Comparator terminology.** The economic comparator is a path-traced re-render with the same frozen ER (comparison 3 of the mandatory baseline g ...[cut]... /8dna26/.venv`, `cuda_ad_rgb` with the OptiX path from `env.sh`. The dev run is `~/rest_x95adev` (not evidenc
docs/worklogs/92A_WORKLOG_ECRO_PHYSICAL_COMPONENT_SUPERVISION.md:14:- **Terminology.** Path tracing and explicit transport are physical reference / null comparator only. No external neural baseline; no Neural Full Recompute claim.
docs/worklogs/92_WORKLOG_ECRO_SHARP_REPRESENTABILITY_IDENTIFIABILITY_ATTRIBUTION.md:12:- **Terminology.** Explicit transport and path tracing are physical comparators and diagnostics. No external neural baseline is run; no Neural Full Recompute claim is made.
RUN [Check baseline-terminology gate usage in WL86-96A protocols] cd /c/Projects/ReST-GI && grep -l -i "neural full recompute" experiments/*/protocol/*.json 2>/dev/null | head -30; echo ---; grep -o -i ".\{120\}neural full recompute.\{160\}" experiments/gm1_explicit87/protocol/*.json experiments/ecro86/protocol/ecro86.json experiments/ecro_util96a/protocol/*.json 2>/dev/null | head -8; echo ---; grep -n -i "neural full recompute\|UNMEASURED\|external" docs/worklogs/86_*.md docs/worklogs/87_*.md docs/worklogs/96A_*.md docs/worklogs/95A_*.md docs/worklogs/92*.md
RUN [Read WL93B/93C interpretations and WL93A scale notes] sed -n '234,251p' docs/worklogs/93B_WORKLOG_ECRO_SOURCE_RECEIVER_FACTOR_SEPARATION_SMOKE.md | cut -c1-1100; echo ===93C; sed -n '185,207p' docs/worklogs/93C_WORKLOG_ECRO_DUAL_STAGE_PHYSICAL_SUPERVISION_SMOKE.md | cut -c1-1100; echo ===93A; grep -n "NO_SIGNAL\|budget\|steps\|smoke" docs/worklogs/93A_*.md | cut -c1-260 | head -14
  -> ## Interpretation

At smoke scale and on the validation population, source / receiver factor separation removes most of the whole-transport cost that source-stage supervision causes in the tied operator, while keeping the source-stage gain. By the frozen mapping this is a **SHARING_SIGNAL**, consist ...[cut]... 7:- **Verdict mapping:** FEASIBLE_FOR_FULL_SCALE_EVALUATION / NO_SIGNAL_AT_SMOKE_SCALE / TARGET_UNRELIABLE / INCONCLUSIVE. These are feasibility assessments only.
87:- Target: T_QGEN (B0 + WL87 seed reproduces the stored E2_T1 q; every budget runs), T_SEEDS.
93:**Dev chain** (`~/rest_x93adev`, `--dev`: Mq 64, 24 / 3 / 6 configurations, 100 steps, 64 spp). Every stage ran: configurations, references, stage tables, Task A, targets, tests, training, evaluation, render tables, renders, analysis, visuals.
101:- **qtable cost per budget.** On one training geometry (timing only), see the table above.
RUN [Read WL93B/93C interpretations and WL93A scale notes] cd /c/Projects/ReST-GI && sed -n '234,251p' docs/worklogs/93B_WORKLOG_ECRO_SOURCE_RECEIVER_FACTOR_SEPARATION_SMOKE.md | cut -c1-1100; echo ===93C; sed -n '185,207p' docs/worklogs/93C_WORKLOG_ECRO_DUAL_STAGE_PHYSICAL_SUPERVISION_SMOKE.md | cut -c1-1100; echo ===93A; grep -n "NO_SIGNAL\|budget\|steps\|smoke" docs/worklogs/93A_*.md | cut -c1-260 | head -14
RUN [Check living docs for WL93-96A references and roadmap header] grep -n -o -E "WL9[3-6]A?|NO_VIABLE|resolvent|M64" docs/Architecture.md docs/RESEARCH_CENTRIC_TOPIC.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md AGENTS.md | sort | uniq -c | head; sed -n 1,12p docs/RESEARCH_ROADMAP.md | cut -c1-900; grep -n "ECRO\|GM1" docs/RESEARCH_CENTRIC_TOPIC.md | cut -c1-300 | head
  -> 1 AGENTS.md:7:WL93
# Research Roadmap — Dynamic Learned Transport Reuse

**Current as of:** 2026-11-13. **Role:** milestone dependencies, current bounded scope and next decision gate. [Research objective](RESEARCH_CENTRIC_TOPIC.md), [architecture contracts and candidate status](Architecture.md), and ...[cut]... 
29:**Current global development decision (human, 2026-11-13):** The ECRO lineage is restored as the **central learned-global-transport implementation base** for GM1, while the WL84 learned ER closure remains the local-scattering development baseline. This selects the research implementation starting
31:**Earlier local-scattering decision (human decision, 2026-11-10; retained):** The **[WL84-qualified learned ER local material closure](worklogs/84_WORKLOG_RENDERING_QUALIFICATION_UTILITY_CLOSURE.md)** (building on WL82–83) becomes the **primary development baseline for local scattering**, not t
RUN [Check living docs for WL93-96A references and roadmap header] cd /c/Projects/ReST-GI && grep -n -o -E "WL9[3-6]A?|NO_VIABLE|resolvent|M64" docs/Architecture.md docs/RESEARCH_CENTRIC_TOPIC.md docs/BASELINE_ROLES_AND_EVIDENCE_STRATEGY.md AGENTS.md | sort | uniq -c | head; sed -n 1,12p docs/RESEARCH_ROADMAP.md | cut -c1-900; grep -n "ECRO\|GM1" docs/RESEARCH_CENTRIC_TOPIC.md | cut -c1-300 | head
RUN [Read GM milestone table and research topic decision] sed -n '13,22p' docs/RESEARCH_ROADMAP.md | cut -c1-1500; sed -n 29,30p docs/RESEARCH_CENTRIC_TOPIC.md | cut -c1-1500
  -> | Milestone | Central question and bounded scope | Evidence required for a decision | Exit / dependency |
|---|---|---|---|
| **GM0 — local foundation** | Can one compact learned, physically admissible local scattering rule serve as the development starting point? | WL82 direct-fit representation, W ...[cut]... lisation verdict. Preserve WL86/86A failure attribution and WL87–WL90 explicit transport, sampling and projection evidence as controls; EXP/C3UR are explicit null/diagnostic comparators, not replacements for the neural-transport problem definition. Any ECRO-lineage successor must demonstrate meaningful learned ownership of global/higher-order transport and satisfy the joint fidelity, edit-validity and quality-matched economic objective. The particular repair/representation hypothesis remains open and requires separate approval and prospective tests; this decision authorises no new agent batch.
RUN [Read GM milestone table and research topic decision] cd /c/Projects/ReST-GI && sed -n '13,22p' docs/RESEARCH_ROADMAP.md | cut -c1-1500; sed -n 29,30p docs/RESEARCH_CENTRIC_TOPIC.md | cut -c1-1500
RUN [Count edit-response measurements in WL93A-96A worklogs] for f in 93A 93B 93C 93D_WORKLOG_ECRO 94_ 94A 94B 94C 94D 95A 96A; do n=$(grep -c -i -E "edit slope|edit response|T0 ?/ ?T1|T0↔T1|signed change|STALE" docs/worklogs/${f}*.md | head -1); echo "$f: $n"; done; grep -n -i -E "edit slope|edit response" docs/worklogs/93A_*.md docs/worklogs/93B_*.md docs/worklogs/93C_*.md docs/worklogs/94*.md docs/worklogs/95A_*.md | cut -c1-220 | head -20
  -> 93A: 4
93B: 3
93C: 1
93D_WORKLOG_ECRO: 3
94_: 0
94A: 0
94B: 0
94C: 0
94D: 0
95A: 0
96A: 0
docs/worklogs/93B_WORKLOG_ECRO_SOURCE_RECEIVER_FACTOR_SEPARATION_SMOKE.md:206:- **Gates and dynamics.** UQ: every WL86 gate passed by T0 is kept, F3 PASS on E2 / E5, no stationary / D regression beyond 1.25 × T0 + 0.002
RUN [Count edit-response measurements in WL93A-96A worklogs] cd /c/Projects/ReST-GI && for f in 93A 93B 93C 93D_WORKLOG_ECRO 94_ 94A 94B 94C 94D 95A 96A; do n=$(grep -c -i -E "edit slope|edit response|T0 ?/ ?T1|T0↔T1|signed change|STALE" docs/worklogs/${f}*.md | head -1); echo "$f: $n"; done; grep -n -i -E "edit slope|edit response" docs/worklogs/93A_*.md docs/worklogs/93B_*.md docs/worklogs/93C_*.md docs/worklogs/94*.md docs/worklogs/95A_*.md | cut -c1-220 | head -20
RUN [Inspect WL87 stage-2 realization json] python - <<'EOF'
import json,re
d=json.load(open('results/gm1_explicit87/results/stage2_87.json'))
def find(o,path='',out=None):
    if out is None: out=[]
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),out)
    elif isinstance(o,list):
        if len(o)>0 and all(isinstance(x,(int,float)) for x in o): out.append((path,o))
        else:
            for i,v in enumerate(o): find(v,path+'/'+str(i),out)
    else: out.append((path,o))
    return out
out=find(d)
print
  -> 5750
['C_RECEIVER_SAMPLING', 'family', 'per_edit']
/per_edit/E1/EXP/rows/0/seed 86001
/per_edit/E1/EXP/rows/0/T0/stationary_all/E_H 0.0038106490118021154
/per_edit/E1/EXP/rows/0/T0/stationary_all/signed -0.0012015509886033455
/per_edit/E1/EXP/rows/0/T0/stationary_all/noise_bound 0.0034659075683 ...[cut]... 939772928005194
/per_edit/E1/EXP/rows/1/T0/C/E_H 0.014302369093428385
/per_edit/E1/EXP/rows/1/T0/C/signed -0.011858733636433067
/per_edit/E1/EXP/rows/1/T0/C/noise_bound 0.006030543518560718
/per_edit/E1/EXP/rows/1/T0/C/capture 0.03578799416816378
/per_edit/E1/EXP/rows/1/T0/D/E_H 0.009869540505365045
/per_edit/E1/EXP/rows/1/T0/D/signed -0.009755813047336836
/per_edit/E1/EXP/rows/1/T0/D/noise_bound 0.0031503283057793675
/per_edit/E1/EXP/rows/1/T0/D/capture 0.044063347009994126
/per_edit/E1/EXP/rows/1/T0/A/E_H 0.01573138168062066
/per_edit/E1/EXP/rows/1/T0/A/signed -0.014838751524083116
RUN [Inspect WL87 stage-2 realization json] cd /c/Projects/ReST-GI && python - <<'EOF'
import json,re
d=json.load(open('results/gm1_explicit87/results/stage2_87.json'))
def find(o,path='',out=None):
    if out is None: out=[]
    if isinstance(o,dict):
        for k,v in o.items(): find(v,path+'/'+str(k),out)
    elif isinstance(o,list):
        if len(o)>0 and all(isinstance(x,(int,float)) for x in o): out.append((path,o))
        else:
            for i,v in enumerate(o): find(v,path+'/'+str(i),out)
    else: out.append((path,o))
    re
NOTE: Now I will write my audit script (numpy only) that reproduces the cheap raw-number checks, then the report.
RUN [Write and run the F0 quote-check script] mkdir -p experiments/audit97/F0 results/audit97/F0 docs/audits/wl97_evidence_reliability && ls docs/audits/wl97_evidence_reliability results/audit97 experiments/audit97 && cat > experiments/audit97/F0/f0_quote_checks.py <<'EOF'
"""WL97 audit, track F0: cheap raw-number checks behind the inference-chain review.

Read-only on every existing file. numpy only. Writes results/audit97/F0/f0_quote_checks.json.
Run from the repository root:  python experiments/audit97/F0/f0_quote_checks.py
"""
import ha
  -> Exit code 2
/usr/bin/bash: -c: line 121: unexpected EOF while looking for matching `''
NOTE: Now I will write my audit script (numpy only) that reproduces the cheap raw-number checks, then the report.
RUN [Write and run the F0 quote-check script] cd /c/Projects/ReST-GI && mkdir -p experiments/audit97/F0 results/audit97/F0 docs/audits/wl97_evidence_reliability && ls docs/audits/wl97_evidence_reliability results/audit97 experiments/audit97 && cat > experiments/audit97/F0/f0_quote_checks.py <<'EOF'
"""WL97 audit, track F0: cheap raw-number checks behind the inference-chain review.

Read-only on every existing file. numpy only. Writes results/audit97/F0/f0_quote_checks.json.
Run from the repository root:  python experiments/audit97/F0/f0_quo
WRITE c:\Projects\ReST-GI\experiments\audit97\F0\f0_quote_checks.py
RUN [Run the F0 quote-check script] mkdir -p results/audit97/F0 docs/audits/wl97_evidence_reliability && python experiments/audit97/F0/f0_quote_checks.py 2>&1 | tail -150
  -> wrote c:\Projects\ReST-GI\results\audit97\F0\f0_quote_checks.json
WL86_E2_C_vs_rule {
 "ecro_s8601|E2|T0|C": {
  "E_H": 0.060280104370449034,
  "noise_bound": 0.01013178292799979,
  "margin_over_0.06": 0.0002801043704490358,
  "margin_in_noise_bounds": 0.02764610853189033
 },
 "ecro_s8601|E2 ...[cut]... 20646199, 'draws_positive': 55, 'n': 64}
WL96A {"G8601/XOD": 39.394592298293134, "G8601/XD": 8.76462068307339, "G8602/XOD": 19.52670069453231, "G8602/XD": 4.344355780701826} {"N": 1270, "R0_edit_wall_ms": 144.13186814635992, "R0_edit_gpu_kernel_ms": 25.467775692930445, "R0_edit_gpu_kernel_share_of_wall": 0.17669774228603613, "R0_source_wall_ms": 88.53004220873117, "tail_share_edit": 0.022973607301324438, "tail_share_source": 0.03739797496505543, "render_share_edit": 0.4129077580982191, "q_table_share_source": 0.2752989690791059, "arith_ceiling_source_if_q_and_sweeps_free": 1.5057021384306095}
WRITE c:\Projects\ReST-GI\experiments\audit97\F0\f0_quote_checks.py
RUN [Run the F0 quote-check script] cd /c/Projects/ReST-GI && mkdir -p results/audit97/F0 docs/audits/wl97_evidence_reliability && python experiments/audit97/F0/f0_quote_checks.py 2>&1 | tail -150
EDIT c:\Projects\ReST-GI\experiments\audit97\F0\f0_quote_checks.py
RUN [Rerun the check script and inspect WL87 ECRO realization rows] python experiments/audit97/F0/f0_quote_checks.py 2>&1 | sed -n '28,80p'; python - <<'EOF'
import json
d=json.load(open('results/gm1_explicit87/results/stage2_87.json'))
print(list(d['per_edit'].keys()), list(d['per_edit']['E2'].keys()))
print([r.get('seed') for r in d['per_edit']['E2']['ECRO']['rows']])
print({k:v for k,v in d['per_edit']['E2']['ECRO'].items() if k!='rows'})
EOF
  -> WL87_E2_T1_C_realizations {
 "EXP": {
  "n": 17,
  "E_H_mean": 0.04468586482671853,
  "E_H_sd": 0.01724976569377454,
  "E_H_min": 0.023738064498964772,
  "E_H_max": 0.07677325514199843,
  "pass_rate_le_0.06": 0.7647058823529411,
  "signed_mean": -0.025667432424640067,
  "first_row_seed": 86001,
  "f ...[cut]...  0.054104913149315564]], 'T1:obj_panel:signed': [-0.03904291326030655, 0.009791189395717596, [-0.0442591694108751, -0.033826657109738006]], 'T1:obj_panel:frac_pass_0.06': 0.8125, 'dH:C:slope': [0.6275549700843985, 0.04408662290118842, [0.6040678217337904, 0.6510421184350067]], 'dH:C:corr': [0.9278504009102218, 0.04021818156921123, [0.9064241646792245, 0.949276637141219]], 'dH:D:slope': [0.795921837385444, 0.014480718145700039, [0.7882072347933223, 0.8036364399775656]], 'dH:D:corr': [0.9694753241211326, 0.005211058919318522, [0.9666991324818657, 0.9722515157603995]], 'frac_F1_F2_pass': 0.5625}}
EDIT c:\Projects\ReST-GI\experiments\audit97\F0\f0_quote_checks.py
RUN [Rerun the check script and inspect WL87 ECRO realization rows] cd /c/Projects/ReST-GI && python experiments/audit97/F0/f0_quote_checks.py 2>&1 | sed -n '28,80p'; python - <<'EOF'
import json
d=json.load(open('results/gm1_explicit87/results/stage2_87.json'))
print(list(d['per_edit'].keys()), list(d['per_edit']['E2'].keys()))
print([r.get('seed') for r in d['per_edit']['E2']['ECRO']['rows']])
print({k:v for k,v in d['per_edit']['E2']['ECRO'].items() if k!='rows'})
EOF
RUN [Check whether ECRO realization pass rate is reported; verify file endings and git status] grep -n -E "56 ?%|0\.56|56\.25|9 ?/ ?16" docs/worklogs/87_*.md docs/RESEARCH_ROADMAP.md | cut -c1-260 | head; file experiments/audit97/F0/f0_quote_checks.py; sha256sum results/audit97/F0/f0_quote_checks.json experiments/audit97/F0/f0_quote_checks.py | cut -c1-80; git -C . status --short | head
  -> docs/RESEARCH_ROADMAP.md:138:- [WL90](worklogs/90_WORKLOG_GM1_INTERMEDIATE_PROJECTION_GAP_ATTRIBUTION.md) (2026-11-12): diagnostic attribution of the common fidelity gap between converged explicit relations and the oracle table on the WL89 workload (W1 T0 / T1
docs/RESEARCH_ROADMAP.md:143:- [WL93](worklogs/93_WORKLOG_ECRO_SOURCE_STAGE_IDENTIFIABILITY_AND_FACTOR_SHARING.md) (2026-11-13): regular numbered batch testing source-stage supervision and source / receiver local-factor sharing on ECRO (global branch frozen to
experiments/audit97/F0/f0_quote_checks.py: Python script, ASCII text executable
92c0769476aad84a4b467b9631a40467c3b0e7d7e0045141dc26e0fe6bf488b2 *results/audit9
ed67f42c933ced67ee4bae2a833112e7e18e4190a7c233b894e76fb7deee5fd2 *experiments/au
?? experiments/audit97/
RUN [Check whether ECRO realization pass rate is reported; verify file endings and git status] cd /c/Projects/ReST-GI && grep -n -E "56 ?%|0\.56|56\.25|9 ?/ ?16" docs/worklogs/87_*.md docs/RESEARCH_ROADMAP.md | cut -c1-260 | head; file experiments/audit97/F0/f0_quote_checks.py; sha256sum results/audit97/F0/f0_quote_checks.json experiments/audit97/F0/f0_quote_checks.py | cut -c1-80; git -C . status --short | head
RUN [Collect line numbers for evidence citations] g(){ grep -n -m2 -E "$1" "$2" | cut -c1-110; }; 
g "GENERALIZATION \| \*\*NOT SUPPORTED|GM1 ARCHITECTURE VIABILITY" docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
g "ECRO \(같은 state\)|realization 통과율" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
g "WL86 gates F1–F4 pass for every model|X_EXP 0.030" docs/worklogs/92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md
g "UNDERREPRESENTED|4 / 96|4.3 %" docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_EXPOSURE_SMOKE.md
g "Explic
  -> 268:| GENERALIZATION | **NOT SUPPORTED** (predeclared mapping) | G1, F1, F4 hold for both seeds; F2 fails for 
272:| GM1 ARCHITECTURE VIABILITY | **NOT SUPPORTED under the frozen gates** | Cause: learned-operator error on
248:| ECRO (같은 state) | 0.062, 0.012 | −0.058 [−0.065, −0.051] | −0.070 | — |  ...[cut]... resentability is not settled: the operator error is far above the known rank-16
163:- Whether a longer or differently weighted teacher fit (not authorised: no sweep or ex
38:| Box A (checker) | centre (−1.08, 0.60, 1.00), half sizes (0.20, 0.45, 0.45), yaw 0.
43:**Scene iteration.** Every step was checked on an actual render (`checker93d.py probe`:
49:| p03 / p04 | largest face, yaw 0.15, face at about 0.22 m | the 3 × 3 checker is clea
52:At p03 / p04 the reference LOW ratio in the reflection is 0.79 / 1.36, against the idea
181:| `scene_iteration/p01–p05*.png` | the scene-iteration renders |
RUN [Collect line numbers for evidence citations] cd /c/Projects/ReST-GI && g(){ grep -n -m2 -E "$1" "$2" | cut -c1-110; }; 
g "GENERALIZATION \| \*\*NOT SUPPORTED|GM1 ARCHITECTURE VIABILITY" docs/worklogs/86_WORKLOG_ECRO_GLOBAL_RESPONSE.md
g "ECRO \(같은 state\)|realization 통과율" docs/worklogs/87_WORKLOG_GM1_EXPLICIT_TRANSPORT_COMPARATOR.md
g "WL86 gates F1–F4 pass for every model|X_EXP 0.030" docs/worklogs/92B_WORKLOG_ECRO_OWN_SHARP_AND_SOURCE_ATTRIBUTION.md
g "UNDERREPRESENTED|4 / 96|4.3 %" docs/worklogs/94B_WORKLOG_ECRO_SHARP_PANEL_TRANSPORT_E
WRITE c:\Projects\ReST-GI\docs\audits\wl97_evidence_reliability\F0_inference_chain.md
RUN [Verify WL92A training steps and the kernel-time field semantics] grep -n -E "24,000|steps" docs/worklogs/92A_*.md | cut -c1-200 | head -8; grep -n -E "gpu_kernel|kernel_history|kernel time" experiments/ecro_util96a/w96a.py | head -8
  -> 26:**Training.** The train86 contract: Adam, lr 1e-3, warm-up 500, cosine to 1e-5, 24,000 steps, one geometry (4 lightings) per step, gradient clip 10; init and data order from the seed. The initial s
68:- **Dev chain** (`~/rest_x92adev`; 64-spp references and renders, 32-spp targets, 48 training ge ...[cut]... hput (dev, four trainings concurrently):** about 100 ms per step for both objectives. Hence about 40 min per training at 24,000 steps; OWN costs no measurable extra time per step. The fu
495:    orig = dr.kernel_history
505:    dr.kernel_history = kh
537:        dr.kernel_history()
559:                    'cpu_s': time.process_time() - self.c0, 'gpu_kernel_ms': self.exec_ms, 'parts': dict(self.parts), 'parts_sum_s': sum(self.parts.values())}
644:                    'gpu_kernel_ms': med(rows, 'gpu_kernel_ms'), 'parts': {k: float(np.median([r['parts'].get(k, 0.0) for r in rows])) for k in keys},
RUN [Verify WL92A training steps and the kernel-time field semantics] cd /c/Projects/ReST-GI && grep -n -E "24,000|steps" docs/worklogs/92A_*.md | cut -c1-200 | head -8; grep -n -E "gpu_kernel|kernel_history|kernel time" experiments/ecro_util96a/w96a.py | head -8
NOTE: You've hit your 5-hour limit · resets 7:10am (Asia/Seoul)
