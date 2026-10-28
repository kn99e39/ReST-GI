"""Machine-readable gates and preliminary server report; no post-hoc threshold choice."""
import sys,os,json,csv
import numpy as np
from r57_data import ROOT,OUT,dump,sha

def main():
    gate=json.load(open(OUT/'gate1.json'));held=json.load(open(OUT/'heldout.json')) if (OUT/'heldout.json').exists() else {};cost=json.load(open(OUT/'cost.json')) if (OUT/'cost.json').exists() else {}
    comp=json.load(open(OUT/'composition.json')) if (OUT/'composition.json').exists() else {}
    out=dict(data_gate=gate['proceed'],useful_models={k:v['useful_and_physical'] for k,v in held.items()},comparison=[],composition=comp.get('executed',False),
        primary_commit=(ROOT/'repo_commit.txt').read_text().strip(),checkpoints={p.name:sha(p) for p in (ROOT/'checkpoints').glob('*.pt') if 'progress' not in p.name})
    out['execution_attempts']={p.stem:json.load(open(p)) for p in sorted((ROOT/'logs').glob('*.attempt*.json'))}
    out['actual_stage_wall_including_failures']={name:sum(r['wall_seconds'] for r in out['execution_attempts'].values() if r['stage']==name) for name in {r['stage'] for r in out['execution_attempts'].values()}}
    if held:
        for pose,cs in cost.items():
            for seed in (5701,5702):
                d,r=cs[f'D_{seed}'],cs[f'R_{seed}'];dd=held[f'D_{seed}'];rr=held[f'R_{seed}']
                qok=all(max(b-a for a,b in zip(dd['poses'][p]['metrics']['coarse']['mean'],rr['poses'][p]['metrics']['coarse']['mean']))<=.02 for p in dd['poses'])
                ratios={n:r['queries'][n]['best']/d['queries'][n]['best'] for n in d['queries']};wins=[]
                for n in ratios:
                    dr=d['queries'][n];rt=r['queries'][n];dl=dr[dr['selected']+'_lifecycle'];rl=rt[rt['selected']+'_lifecycle'];base=d['geometry']['median']
                    if ratios[n]<=.75 and base+rl['p90']<base+dl['p10']:wins.append(int(n))
                # Same chosen full/compact policy includes geometry and coefficient state.
                def memory(row):
                    return max(row['cache_policy_total_bytes'][row['queries'][n]['selected']] for n in row['queries'])
                memratio=memory(r)/memory(d)
                supported=dd['useful_and_physical'] and rr['useful_and_physical'] and qok and len(wins)>=2 and any(n in wins for n in (1,4)) and memratio<=1.25 and r['parameters']/d['parameters']<=1.05
                out['comparison'].append(dict(pose=pose,seed=seed,cost_ratios=ratios,memory_ratio=memratio,quality_within_02=qok,qualified_query_counts=wins,R_advantage_supported=supported))
        useful=any(out['useful_models'].values());out['verdict']=dict(
            direct_viability='SUPPORTED' if all(held[f'D_{s}']['useful_and_physical'] for s in (5701,5702)) else 'NOT SUPPORTED',
            rule_state_viability='SUPPORTED' if all(held[f'R_{s}']['useful_and_physical'] for s in (5701,5702)) else 'NOT SUPPORTED',
            rule_state_advantage='SUPPORTED' if out['comparison'] and all(r['R_advantage_supported'] for r in out['comparison']) else ('NOT SUPPORTED' if not useful else 'INCONCLUSIVE'),
            physical_contract='SUPPORTED' if all(p['physics']['passed'] for v in held.values() for p in v['poses'].values()) else 'NOT SUPPORTED',
            learned_K_readiness='CONDITIONALLY SUPPORTED' if useful else 'NOT SUPPORTED',
            multi_domain_composition='INCONCLUSIVE' if not comp.get('executed') else 'See separately reported G5 comparisons; oracle floor retained',
            global_and_contact='INCONCLUSIVE',TransGI_comparison='INCONCLUSIVE')
        rows=[]
        for tag,v in held.items():
            for pose,p in v['poses'].items():
                for resolution in ('coarse','fine16','fine64'):
                    for c,ch in enumerate('RGB'):rows.append(dict(model=tag,pose=pose,resolution=resolution,channel=ch,mean_relative_L2=p['metrics'][resolution]['mean'][c],p90_relative_L2=p['metrics'][resolution]['p90'][c],static_useful=p['static_useful'],physics=p['physics']['passed']))
        with open(OUT/'operator_metrics.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    else:out['verdict']={'data_readiness':'NOT SUPPORTED','learned_architecture':'INCONCLUSIVE','reason':'Preregistered data gate failed; no primary training.'}
    dump(out,OUT/'verdict.json')
    (OUT/'SERVER_REPORT.md').write_text('# WL57 server chain report\n\nPrimary commit: `'+out['primary_commit']+'`\n\n```json\n'+json.dumps(out.get('verdict'),indent=2)+'\n```\n\nFull measurements: heldout.json, cost.json, gate1.json, composition.json, operator_metrics.csv. Scope: local dent/rigid material frame only; no architecture adoption or WL58.\n')
    manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'};dump(manifest,OUT/'artifact_manifest.json')
    print(json.dumps(out,indent=1))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
