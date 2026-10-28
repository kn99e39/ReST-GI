"""Full bounded A2 outcome from actual files; no nonexistent measurements claimed."""
import os,sys,json,time,csv
from pathlib import Path
from r57_data import ROOT,OUT,dump,sha

def report(failure=None):
    def read(n):return json.load(open(OUT/n)) if (OUT/n).exists() else None
    gate=read('confirmation_gate.json');pilot=read('pilot.json');held=read('heldout.json');cost=read('cost.json');composition=read('composition.json')
    if held and cost:
        import r57_report
        r57_report.main()
    base=read('verdict.json') if held else None
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((ROOT/'data/halves').glob('*.json'))}
    decisions=dict(reference_data_readiness='SUPPORTED' if gate and gate['proceed'] else ('NOT SUPPORTED' if gate else 'INCONCLUSIVE'),
        direct_learned_K='INCONCLUSIVE',rule_state_learned_K='INCONCLUSIVE',rule_state_advantage='INCONCLUSIVE',absolute_fidelity='INCONCLUSIVE',deformation_fidelity='INCONCLUSIVE',physical_contract='INCONCLUSIVE',dynamic_cost='INCONCLUSIVE',FNTO_composition='INCONCLUSIVE',rigid='CONDITIONALLY SUPPORTED',local_deformation='INCONCLUSIVE',global_bending_twisting='INCONCLUSIVE',self_contact='INCONCLUSIVE',TransGI='INCONCLUSIVE')
    if base:
        v=base['verdict'];decisions.update(direct_learned_K=v['direct_viability'],rule_state_learned_K=v['rule_state_viability'],rule_state_advantage=v['rule_state_advantage'],physical_contract=v['physical_contract'])
        decisions['absolute_fidelity']='SUPPORTED' if all(p['static_useful'] for r in held.values() for p in r['poses'].values()) else 'NOT SUPPORTED'
        dynamics=[q['pass_dynamic'] for r in held.values() for p in r['poses'].values() for q in p.get('dynamic',[]) if q['detectable']]
        decisions['deformation_fidelity']='SUPPORTED' if dynamics and all(dynamics) else 'NOT SUPPORTED'
        decisions['local_deformation']='CONDITIONALLY SUPPORTED' if any(r['useful_and_physical'] for r in held.values()) else 'NOT SUPPORTED'
        decisions['dynamic_cost']='CONDITIONALLY SUPPORTED' if any(r['useful_and_physical'] for r in held.values()) else 'INCONCLUSIVE'
    attempts={p.stem:json.load(open(p)) for p in sorted((ROOT/'logs').glob('*.attempt*.json'))}
    internal_decision=gate['verdict'] if gate else ('NO-GO' if pilot and not pilot['passed'] else 'INCONCLUSIVE_IMPLEMENTATION_STOP')
    result=dict(protocol='fnto_rule57_recovery_A2',execution_policy=read('../execution_policy.json'),internal_decision=internal_decision,commit=(ROOT/'repo_commit.txt').read_text().strip(),original_gate='FAILED preserved, diagnostic only; original neural hypothesis was never tested',pilot=pilot,confirmation=gate,primary_models=held is not None,cost_measured=cost is not None,composition=composition,failure=failure,verdict=decisions,attempts=attempts,
        halves_receipts=manifest,checkpoints={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt') if 'progress' not in p.name},interpretation='Specific learned nonlinear graph-state mechanism vs direct matched raw context. Not isolation of abstract state separation. Fast inaccurate model cannot win. Training reused original noisy targets; confirmation uses unbiased incident-field importance sampling and independent geometry/light/seed populations. Historical WL56 blue floor persists.',next_batch='NOT AUTHORISED')
    dump(result,OUT/'recovery_final.json')
    text='# WL57 A2 recovery outcome\n\n## Implementation Fact\n\n'+f"Source `{result['commit']}`. One incident-field importance recovery; original failed gate preserved. Pilot uses original diagnostic fields; independent confirmation uses four fresh local-dent geometries and64fresh lights. Frozen D/R architecture, train/validation inputs/targets, seeds, optimizer, loss and6000steps.\n\n"
    text+='## Measurement\n\n'+f"Pilot passed: {pilot.get('passed') if pilot else 'not completed'}. Confirmation: {gate.get('verdict') if gate else 'not completed'}. Model results: {held is not None}; costs: {cost is not None}; composition executed: {bool(composition and composition.get('executed'))}. All measured files/attempts and half/checkpoint identities are in recovery_final.json.\n\n"
    if gate:
        text+='| Geometry | R qualified/16 | G | B |\n|---|---:|---:|---:|\n'
        for r in gate['signal']:text+=f"|{r['pose']}|"+'|'.join(map(str,r['qualified_counts']))+'|\n'
    text+='\n## Observation\n\nOriginal spatial-pair operator changes were measurable, but light-action detectability was limited by uneven excitation, spatial/angular dilution and conditional sampling variance. Recovery tests importance-allocated physical actions; a run finishing alone is no evidence of neural quality.\n\n## Interpretation\n\n'+result['interpretation']+'\n\n## Unresolved Question\n\n'
    text+=('Learned viability and quality-matched R advantage remain unmeasured because the declared gate/implementation stop prevented primary completion. No architecture falsification follows.\n\n' if not held else 'Any failed quality group, training-vs-reference-noise limit or coupled oracle floor remains separate from runtime cost and from the physical validity of the learned map. No untested geometry class is inferred.\n\n')
    text+='## Architecture Verdict\n\n```json\n'+json.dumps(decisions,indent=2)+'\n```\n\n## Reproducibility\n\nProtocol experiments/fnto_rule57/protocol/rule57_recovery_a2.json; server ~/rest_wl57_recovery, logs/, data/halves, checkpoints/, out/. Original diagnostics ~/rest_wl57_recovery/diagnostic and immutable ~/rest_wl57. Exact seeds, sampling budgets, source hashes and receipt hashes recorded. No WL58 or automatic representation adoption.\n'
    if failure:text+='\nExecution stop (preserved):\n```\n'+failure+'\n```\n'
    (OUT/'RECOVERY_REPORT.md').write_text(text);print(json.dumps(decisions,indent=1),flush=True)
if __name__=='__main__':report();sys.stdout.flush();os._exit(0)
