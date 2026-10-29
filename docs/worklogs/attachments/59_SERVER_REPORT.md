# Persistent Intrinsic F/S — server completion report

Status: COMPLETE

## Implementation Fact
One dependency-local positive graph rule; same checkpoint for F and actual resident S. Nonrigid B and global normalization rebuilt and charged. No source/target lookup in state construction.

## Measurement
### P_5901
- pt0 coarse mean RGB F: [0.0615256205201149, 0.06200031936168671, 0.0667659267783165]
- fine16 mean RGB F: [0.10200382769107819, 0.10173016786575317, 0.11148443073034286]
- pt0 coarse mean RGB S: [0.0615256205201149, 0.06200031936168671, 0.0667659193277359]
- fine16 mean RGB S: [0.10200382769107819, 0.10173016786575317, 0.11148443073034286]
- F/S max relative L1: 6.734126145602204e-08; retained raw: 0.6611307779947917
- Dynamic F: [{'detectable': True, 'n_fields': 11, 'error': 0.19081413745880127, 'corr': 0.9822034239768982, 'slope': 0.9316450357437134, 'passed': True}, {'detectable': True, 'n_fields': 10, 'error': 0.1897333264350891, 'corr': 0.9819176197052002, 'slope': 0.9490259289741516, 'passed': True}, {'detectable': True, 'n_fields': 8, 'error': 0.410084992647171, 'corr': 0.9133917093276978, 'slope': 0.8874173760414124, 'passed': False}]
- Directional F: {'errors': [[0.25425875186920166, 0.19426387548446655, 0.11114738136529922], [0.24395926296710968, 0.24697273969650269, 0.17441286146640778], [0.24561598896980286, 0.2141137421131134, 0.17327454686164856], [0.2097282111644745, 0.20338837802410126, 0.16988205909729004], [0.24892988801002502, 0.25024107098579407, 0.20707769691944122], [0.24347688257694244, 0.23944091796875, 0.18381786346435547], [0.2021269053220749, 0.2323121279478073, 0.1561136096715927], [0.5070763230323792, 0.4908900856971741, 0.4212802052497864]], 'detectable': [[True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True]], 'qualified': 24, 'passed': False}
### P_5902
- pt0 coarse mean RGB F: [0.06180896237492561, 0.06270422041416168, 0.06636418402194977]
- fine16 mean RGB F: [0.10278545320034027, 0.10408160090446472, 0.11146249622106552]
- pt0 coarse mean RGB S: [0.06180896237492561, 0.06270422041416168, 0.06636417657136917]
- fine16 mean RGB S: [0.10278545320034027, 0.10408160090446472, 0.11146249622106552]
- F/S max relative L1: 9.983440207861349e-08; retained raw: 0.6611307779947917
- Dynamic F: [{'detectable': True, 'n_fields': 11, 'error': 0.20991480350494385, 'corr': 0.9781714677810669, 'slope': 0.9277049899101257, 'passed': True}, {'detectable': True, 'n_fields': 10, 'error': 0.20020033419132233, 'corr': 0.9797950387001038, 'slope': 0.9471004009246826, 'passed': True}, {'detectable': True, 'n_fields': 8, 'error': 0.40869995951652527, 'corr': 0.9146385192871094, 'slope': 0.8983663320541382, 'passed': False}]
- Directional F: {'errors': [[0.2087295800447464, 0.20363849401474, 0.1027023196220398], [0.2356513887643814, 0.24899256229400635, 0.17315340042114258], [0.23439748585224152, 0.19816067814826965, 0.17008453607559204], [0.2091945856809616, 0.20891329646110535, 0.16601282358169556], [0.24149151146411896, 0.2559839189052582, 0.204311341047287], [0.22841040790081024, 0.22348752617835999, 0.183747336268425], [0.21130512654781342, 0.20690134167671204, 0.1405028998851776], [0.4873436391353607, 0.47340840101242065, 0.43491876125335693]], 'detectable': [[True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True]], 'qualified': 24, 'passed': False}
- COST {'seed': 5901, 'edit': 'dent', 'queries': 1, 'F_seconds': 0.3362118471413851, 'S_seconds': 0.3314643744379282, 'ratio': 0.9858795198806292, 'bootstrap95': [0.9700701694915477, 0.9961017420586114], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'dent', 'queries': 4, 'F_seconds': 0.3410707823932171, 'S_seconds': 0.3366399947553873, 'ratio': 0.987009184408175, 'bootstrap95': [0.97091246055438, 0.9962888430375904], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'dent', 'queries': 16, 'F_seconds': 0.35986167192459106, 'S_seconds': 0.3562353942543268, 'ratio': 0.9899231344897877, 'bootstrap95': [0.9738922286687771, 0.9980773205872794], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'dent', 'queries': 64, 'F_seconds': 0.4340657237917185, 'S_seconds': 0.4332181829959154, 'ratio': 0.9980474367144231, 'bootstrap95': [0.9830794329780316, 0.9991670087471548], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'rigid', 'queries': 1, 'F_seconds': 0.36725373938679695, 'S_seconds': 0.26598814874887466, 'ratio': 0.7242626016361187, 'bootstrap95': [0.7118953698264133, 0.7315677200800422], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'rigid', 'queries': 4, 'F_seconds': 0.3722589146345854, 'S_seconds': 0.2710009217262268, 'ratio': 0.7279904149300095, 'bootstrap95': [0.7160345780065324, 0.735815275503323], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'rigid', 'queries': 16, 'F_seconds': 0.39111828804016113, 'S_seconds': 0.2906974907964468, 'ratio': 0.7432469912186698, 'bootstrap95': [0.731639595637287, 0.749670359792304], 'uncontended': True}
- COST {'seed': 5901, 'edit': 'rigid', 'queries': 64, 'F_seconds': 0.46577827632427216, 'S_seconds': 0.3687423262745142, 'ratio': 0.7916692233576774, 'bootstrap95': [0.7782002282714622, 0.7946594378868135], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'dent', 'queries': 1, 'F_seconds': 0.33718223311007023, 'S_seconds': 0.3314178306609392, 'ratio': 0.9829041927981736, 'bootstrap95': [0.9809505217718034, 0.9917660504019115], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'dent', 'queries': 4, 'F_seconds': 0.3422759883105755, 'S_seconds': 0.33639258705079556, 'ratio': 0.9828109436223688, 'bootstrap95': [0.9807467624501792, 0.9913877446939432], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'dent', 'queries': 16, 'F_seconds': 0.3620813023298979, 'S_seconds': 0.35564906895160675, 'ratio': 0.9822353892982008, 'bootstrap95': [0.9795992431781676, 0.9897661165951893], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'dent', 'queries': 64, 'F_seconds': 0.4407161846756935, 'S_seconds': 0.4343152139335871, 'ratio': 0.9854759798603342, 'bootstrap95': [0.9756602032033566, 0.9875847924430589], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'rigid', 'queries': 1, 'F_seconds': 0.3741083014756441, 'S_seconds': 0.26611105166375637, 'ratio': 0.7113208945487173, 'bootstrap95': [0.7026471843085752, 0.7134592868466311], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'rigid', 'queries': 4, 'F_seconds': 0.3792467284947634, 'S_seconds': 0.27130759321153164, 'ratio': 0.7153854544463865, 'bootstrap95': [0.7073202670403014, 0.7176885360327675], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'rigid', 'queries': 16, 'F_seconds': 0.3983370680361986, 'S_seconds': 0.29086711071431637, 'ratio': 0.7302034735263051, 'bootstrap95': [0.7238555685022346, 0.7341391759797784], 'uncontended': True}
- COST {'seed': 5902, 'edit': 'rigid', 'queries': 64, 'F_seconds': 0.4735656473785639, 'S_seconds': 0.36841953732073307, 'ratio': 0.7779693044884693, 'bootstrap95': [0.7752261111837815, 0.7806736501680177], 'uncontended': True}

## Observation
Global state effects are explicitly reduced; preserved raw computation is not confused with unchanged radiance. All diagnostic poses/channels remain in heldout.json.

## Interpretation
Physical learned fidelity and F/S equivalence are separate verdicts. If F misses required fidelity, matching F does not establish physical validity or a quality-matched speedup.

## Unresolved Question
Any remaining representation/supervision/optimization bias is not causally isolated by this single candidate. No successor selected.

## Architecture Verdict
- rule_input_generalization: NOT SUPPORTED
- primary_directional_adequacy: NOT SUPPORTED
- full_state_learned_fidelity: NOT SUPPORTED
- FS_equivalence: SUPPORTED
- invalidation_correctness: SUPPORTED
- selective_update_quality_matched_benefit: NOT SUPPORTED
- actual_coupled_GI_eligibility: NOT SUPPORTED
- sharp_glossy_global_deformation_self_contact: INCONCLUSIVE

## Reproducibility
Source d7256dc
Protocol SHA 1e84a085c0a1916ecefe1a2607d1df30fa22fd2fbe38a2bf02a3bd0b4cc0134b
Seeds5901/5902,6000steps, final checkpoints; physical reference unchanged. Complete stage/attempt logs, data manifests, environment, model and artifact receipts retained under this server root.
Sharp glossy, global bends and self-contact are outside scope. WL58 evidence unchanged. No new hypothesis or production adoption.
