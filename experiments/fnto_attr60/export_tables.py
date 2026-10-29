"""Export review tables from immutable diagnostic JSON, with no filtering of failures."""
import csv,json,hashlib
from pathlib import Path

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
    root=Path(__file__).resolve().parents[2];src=root/'results/fnto_attr60';dest=root/'results/evaluation/60';dest.mkdir(exist_ok=True,parents=True)
    a=json.loads((src/'stored_analysis.json').read_text());l=json.loads((src/'local_summary.json').read_text());r=json.loads((src/'replay.json').read_text())
    bounds=[];allrows=[];replay=[]
    for seed,v in a['seeds'].items():
        for kind,key in [('primary_delta','dynamic'),('primary_directional','directional')]:
            for row in v[key]:
                bounds.append(dict(seed=seed,population=kind,pair=row.get('pair',''),channel=row['channel'],observed_error=row['error'],frozen_output_span_floor=row['frozen_output_span_floor'],fraction_squared_error_outside_frozen_span=(row['frozen_output_span_floor']/row['error'])**2,relaxed_rank8_floor=row.get('free_rank8_floor','')))
        for pose,p in l['seeds'][seed]['poses'].items():
            if 'directional' not in p:continue
            for pair,errs in enumerate(p['directional']['errors']):
                for c,error in enumerate(errs):allrows.append(dict(seed=seed,pose=pose,primary=p['primary'],kind='directional',pair=pair,channel=c,error=error,qualified=p['directional']['detectable'][pair][c]))
            for c,d in enumerate(p['dynamic']):allrows.append(dict(seed=seed,pose=pose,primary=p['primary'],kind='delta',pair='',channel=c,error=d.get('error',''),qualified=d['detectable']))
        for key,p in r['seeds'][seed].items():
            if not isinstance(p,dict):continue
            for c,error in enumerate(p['directional_pair_error']):
                replay.append(dict(seed=seed,pose_light=key,channel=c,original_label_pair_error=error,focused_delta_error_unqualified=p.get('focused_delta_error_unqualified',['','',''])[c]))
    write(dest/'primary_bounds.csv',bounds);write(dest/'all_primary_secondary_failures.csv',allrows);write(dest/'crossed_replay.csv',replay)
    manifest=dict(scope='Post hoc frozen-model diagnostics; not new primary evidence or adopted architecture',source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.json') if p.name!='provenance.json'},artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.csv')})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
