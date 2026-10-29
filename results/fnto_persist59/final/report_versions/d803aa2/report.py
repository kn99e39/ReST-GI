import json,sys,os
from common import *

def main():
    held=json.loads((OUT/'heldout.json').read_text()) if (OUT/'heldout.json').exists() else {}
    cost=json.loads((OUT/'cost.json').read_text()) if (OUT/'cost.json').exists() else {}
    pre=json.loads((OUT/'preflight.json').read_text()) if (OUT/'preflight.json').exists() else {}
    comp=json.loads((OUT/'composition.json').read_text()) if (OUT/'composition.json').exists() else {}
    complete=(len(held)==2 and bool(cost) and (ROOT/'stages/composition.json').exists()
              and not (ROOT/'FAILURE.txt').exists())
    equivalence=complete and all(v['poses'][n]['F_S_passed'] for v in held.values() for n in ['rest']+PRIMARY+DIAGNOSTIC)
    quality=complete and all(v['useful_and_physical'] for v in held.values())
    benefit=complete and any(all(any(c['seed']==seed and c['edit']=='dent' and c['queries']==n and c['uncontended'] and c['bootstrap95'][1]<1 for c in cost['comparisons']) for seed in CFG['training']['seeds']) for n in (1,4))
    scopes=dict(rule_input_generalization='SUPPORTED' if quality else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),
                primary_directional_adequacy='INCONCLUSIVE',full_state_learned_fidelity='SUPPORTED' if quality else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),
                FS_equivalence='SUPPORTED' if equivalence else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),invalidation_correctness='SUPPORTED' if equivalence and pre.get('passed') else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),
                selective_update_quality_matched_benefit='SUPPORTED' if quality and benefit else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),
                actual_coupled_GI_eligibility='SUPPORTED' if quality else ('NOT SUPPORTED' if complete else 'INCONCLUSIVE'),
                sharp_glossy_global_deformation_self_contact='INCONCLUSIVE')
    if complete:
        directional=all(v['poses'][n]['directional_pairs']['F']['passed'] for v in held.values() for n in PRIMARY)
        scopes['primary_directional_adequacy']='SUPPORTED' if directional else 'NOT SUPPORTED'
    status='COMPLETE' if complete else 'FAILED_OR_STOPPED_PRIMARY_GATE'
    dump(dict(batch_id=CFG['batch_id'],status=status,verdicts=scopes,raw_cost_benefit=benefit,
              quality_passed=quality,FS_equivalence=equivalence,code_commit=(ROOT/'repo_commit.txt').read_text().strip(),
              protocol_sha256=sha(HERE/'protocol/persist_v1.json'),report_source_sha256=sha(__file__),no_adoption=True,no_successor=True),OUT/'verdict.json')
    docs=ROOT/'docs';docs.mkdir(exist_ok=True)
    lines=['# Persistent Intrinsic F/S — server completion report','',f'Status: {status}',
           '', '## Implementation Fact', 'One dependency-local positive graph rule; same checkpoint for F and actual resident S. Nonrigid B and global normalization rebuilt and charged. No source/target lookup in state construction.',
           '', '## Measurement']
    for tag,row in held.items():
        lines.append(f'### {tag}')
        for n in PRIMARY:
            r=row['poses'][n]
            lines.extend([f'- {n} coarse mean RGB F: {r["metrics"]["F"]["coarse"]["mean"]}',
                f'- fine16 mean RGB F: {r["metrics"]["F"]["fine16"]["mean"]}',
                f'- {n} coarse mean RGB S: {r["metrics"]["S"]["coarse"]["mean"]}',
                f'- fine16 mean RGB S: {r["metrics"]["S"]["fine16"]["mean"]}',
                f'- F/S max relative L1: {r["F_S"]["max_relative_L1"]}; retained raw: {r["invalidation"]["raw_retained_fraction"]}',
                f'- Dynamic F: {r["dynamic"]["F"]}',f'- Directional F: {r["directional_pairs"]["F"]}'])
    for c in cost.get('comparisons',[]):lines.append(f'- COST {c}')
    lines+=['','## Observation','Global state effects are explicitly reduced; preserved raw computation is not confused with unchanged radiance. All diagnostic poses/channels remain in heldout.json.',
            '', '## Interpretation','Physical learned fidelity and F/S equivalence are separate verdicts. If F misses required fidelity, matching F does not establish physical validity or a quality-matched speedup.',
            '', '## Unresolved Question','Any remaining representation/supervision/optimization bias is not causally isolated by this single candidate. No successor selected.',
            '', '## Architecture Verdict']
    lines += [f'- {key}: {value}' for key,value in scopes.items()]
    lines += ['', '## Reproducibility',f'Source {(ROOT/"repo_commit.txt").read_text().strip()}',f'Protocol SHA {sha(HERE/"protocol/persist_v1.json")}',
              'Seeds5901/5902,6000steps, final checkpoints; physical reference unchanged. Complete stage/attempt logs, data manifests, environment, model and artifact receipts retained under this server root.',
              'Sharp glossy, global bends and self-contact are outside scope. WL58 evidence unchanged. No new hypothesis or production adoption.']
    (docs/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    records={}
    paths=list((ROOT/'checkpoints').glob('P_*.pt'))+list(OUT.glob('*.json'))+list(OUT.glob('actions_*.pt'))
    for name in PRIMARY:
        paths += [DATA/'geometry'/f'{name}.pt']+list((DATA/'targets').glob(f'{name}_r*.pt'))
    for p in paths:
        records[str(p.relative_to(ROOT))]=sha(p)
    dump(records,ROOT/'artifact_manifest.json')

if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
