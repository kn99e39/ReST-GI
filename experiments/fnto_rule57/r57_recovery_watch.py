"""Read-only experiment observer plus terminal report formatting (no science decisions).
Writes its own progress/final-review files only; never starts/restarts/kills a stage.
"""
import os,time,json,statistics,hashlib
from pathlib import Path
ROOT=Path.home()/'rest_wl57_recovery';OUT=ROOT/'out'
def read(p,default=None):
    try:return json.load(open(p))
    except (FileNotFoundError,json.JSONDecodeError):return default
def write(d,p):
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,indent=1)+'\n');os.replace(t,p)
def review():
    final=read(OUT/'recovery_final.json',{});held=read(OUT/'heldout.json',{});cost=read(OUT/'cost.json',{});comp=read(OUT/'composition.json',{})
    s='# WL57 A2 terminal measurement review\n\nObserver formatting only; frozen decisions come from primary gate/evaluation files.\n\n'
    s+='Primary source: `'+(ROOT/'repo_commit.txt').read_text().strip()+'`. Original WL57 Gate1 remains FAILED and diagnostic.\n\n'
    s+='## Declared gate and outcome\n\n```json\n'+json.dumps(final.get('verdict',{}),indent=2)+'\n```\n\n'
    if held:
        s+='## Every fixed model/seed (worst geometry/channel)\n\n|Model|Coarse mean relL2 max|Coarse p90 max|Fine16 mean max|Delta relL2 max|Delta correlation min|Useful+physical|\n|---|---:|---:|---:|---:|---:|---|\n'
        for tag,r in held.items():
            ps=list(r['poses'].values());ds=[x for p in ps for x in p.get('dynamic',[]) if x['detectable']]
            vals=[max(max(p['metrics']['coarse']['mean']) for p in ps),max(max(p['metrics']['coarse']['p90']) for p in ps),max(max(p['metrics']['fine16']['mean']) for p in ps),max((q['relative_L2'] for q in ds),default=float('nan')),min((q['corr'] for q in ds),default=float('nan'))]
            s+='|'+tag+'|'+'|'.join(f'{x:.6g}' for x in vals)+'|'+str(r['useful_and_physical'])+'|\n'
        s+='\nAll geometry/channel/field and angular/noise rows remain in heldout.json and operator_metrics.csv; the worst-case view above does not replace them.\n\n'
    if cost:
        s+='## Fair best-cache lifecycle, including common geometry (seconds)\n\n|Pose|Seed|Queries|D|R|R/D|D policy|R policy|\n|---|---:|---:|---:|---:|---:|---|---|\n'
        for pose,rs in cost.items():
            for seed in (5701,5702):
                d,r=rs[f'D_{seed}'],rs[f'R_{seed}']
                for n in ('1','4','16','64'):
                    a,b=d['queries'][n],r['queries'][n]
                    s+=f"|{pose}|{seed}|{n}|{a['best']:.6g}|{b['best']:.6g}|{b['best']/a['best']:.6g}|{a['selected']}|{b['selected']}|\n"
        s+='\nA lower ratio is not an architecture advantage unless both arms pass all frozen quality/physical conditions and the>=25% noninferiority/memory/timing rule. Oracle data generation and training are offline costs, separately retained in stage attempt logs.\n\n'
    s+='## Conditional composition\n\n'+('Executed known historical WL56 transfer scene, not fresh confirmation. The existing blue oracle-vs-Cycles floor remains separate from learned-vs-oracle error.\n\n' if comp.get('executed') else 'Not executed unless the fixed learned operator gates permit it. No coupled physical-GI success is inferred.\n\n')
    if comp.get('executed'):
        s+='Every failed ruled primary group/channel below; raw passed and failed rows remain in composition.json.\n\n'
        for state,body in comp.get('states',{}).items():
            for tag,model in body['models'].items():
                for row in model['comparisons']:
                    for c,ch in enumerate('RGB'):
                        if row.get('primary') and row['ruled'][c] and not row['passed'][c]:s+=f"- {state}/{tag}/{row['mask']}/{row['cam']}/{row['group']} {ch}: bias {100*row['relative_bias'][c]:.4g}%\n"
    s+='\n## Interpretation limits and four geometry cases\n\nRigid material-frame reuse was tested in A1. Local dents are the sole primary deformation family. Global bending/twisting and self-contact are untested; no sparse/local/low-rank change assumption is validated. A benefit, if present, concerns this concrete learned nonlinear graph-state mechanism against matched-context direct conditioning, not abstract Rule-State labels. Old training targets retain their original Monte Carlo noise; new independent confirmation uses unbiased illumination importance sampling. Distinguish data/reference noise, training/optimization, representation bias, geometry generalization and real lifecycle cost. No TransGI superiority, production adoption or WL58 follows.\n\n'
    if final.get('failure'):s+='## Execution limitation\n\n```\n'+final['failure']+'\n```\n'
    (OUT/'FINAL_REVIEW.md').write_text(s)
    (OUT/'WORKLOG57_A2_FINAL_APPEND.md').write_text('\n\n## WL57 A2 final machine-generated measurement appendix\n\n'+s)
    write(dict(primary_source=(ROOT/'repo_commit.txt').read_text().strip(),observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),measurement_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md') and p.name!='terminal_manifest.json'}),OUT/'terminal_manifest.json')
def main():
    while True:
        status=read(ROOT/'status.json',{});files=list((ROOT/'data/halves').glob('*.json'));rows=[read(p,{}) for p in files];times=[r['seconds'] for r in rows if 'seconds' in r];med=statistics.median(times) if times else None
        try:pid=int((ROOT/'chain.pid').read_text());os.kill(pid,0);alive=True
        except (FileNotFoundError,ProcessLookupError):alive=False
        goal=320 if status.get('stage')=='confirm_focus' else 1280
        write(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),primary=status,supervisor_alive=alive,completed_confirmatory_halves=len(rows),focus_target=320,full_target_if_GO=1280,completed_physical_passes=sum(bool(r.get('physical')) for r in rows),median_seconds_per32M_half=med,rough_remaining_data_seconds=(goal-len(rows))*med if med and len(rows)<goal else None,note='Read-only observer; estimates vary by geometry/light and exclude training/cost; no scientific decision or stage mutation.'),ROOT/'progress.json')
        if status.get('status') in ('COMPLETE','NO_GO_CONFIRMATION','FAILED') or not alive:
            review();return
        time.sleep(30)
if __name__=='__main__':main()
