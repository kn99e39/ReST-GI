"""Read-only reconciliation of existing WL59 primary and WL60/61 diagnostic evidence."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path

def moment_sensitivity(error,noise):
    if error<0 or noise<0:raise ValueError('Norm ratios must be nonnegative')
    if noise>=1:return dict(relative_squared=None,relative_error=None)
    squared=(error*error-noise*noise)/(1-noise*noise)
    return dict(relative_squared=squared,relative_error=math.sqrt(squared) if squared>=0 else None)

def assert_distinct(train,test):
    if set(train)&set(test):raise ValueError('Train and test arrays alias: cannot separate populations')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();root=a.repo
    paths=dict(primary_noise='results/fnto_persist59/final/out/review/noise_sensitivity.json',primary_scores='results/fnto_persist59/final/out/heldout.json',bounds='results/fnto_attr60/stored_analysis.json',history='results/fnto_attr60/local_summary.json',clean='results/fnto_delta61/analysis.json',clean_identity='results/fnto_delta61/preflight.json',clean_review='results/fnto_delta61/noise_review.json',protocol='experiments/fnto_persist59/protocol/persist_v1.json')
    data={k:json.loads((root/p).read_text()) for k,p in paths.items()}
    pop=data['bounds']['light_populations'];train=pop['train']['field_sha256'];test=pop['test']['field_sha256'];assert_distinct(train,test)
    assert data['clean']['fields']==[8,9] and data['clean']['light_split']=='train'
    for f in [8,9]:assert data['clean_identity']['fields'][str(f)]==train[f]
    rows=[]
    for seed,record in data['primary_noise']['seeds'].items():
        bounds=data['bounds']['seeds'][seed]
        for d in record['dynamic']:
            c=d['channel'];error=d['relative_delta_errors_mean_rep0_rep1'][0];noise=d['reference_mean_delta_noise_norm_ratio'];b=bounds['dynamic'][c]
            assert abs(error-b['error'])<1e-5
            rows.append(dict(seed=seed,population='WL59 primary pt0 vs rest; test first16 qualified',kind='delta',pair='',channel=c,qualified_fields=d['n_fields'],raw_error=error,mean_reference_noise_ratio=noise,rep0_error=d['relative_delta_errors_mean_rep0_rep1'][1],rep1_error=d['relative_delta_errors_mean_rep0_rep1'][2],frozen_output_span_floor=b['frozen_output_span_floor'],**moment_sensitivity(error,noise)))
        for pair in range(8):
            for c in range(3):
                error=record['directional_errors_mean_rep0_rep1'][0][pair][c];noise=record['directional_reference_mean_noise_ratio'][pair][c]
                b=next(r for r in bounds['directional'] if r['pair']==pair and r['channel']==c)
                assert abs(error-b['error'])<1e-5
                rows.append(dict(seed=seed,population='WL59 primary pt0; test fields48+2pair/49+2pair',kind='directional',pair=pair,channel=c,qualified_fields=2,raw_error=error,mean_reference_noise_ratio=noise,rep0_error=record['directional_errors_mean_rep0_rep1'][1][pair][c],rep1_error=record['directional_errors_mean_rep0_rep1'][2][pair][c],frozen_output_span_floor=b['frozen_output_span_floor'],**moment_sensitivity(error,noise)))
    a.out.mkdir(parents=True,exist_ok=True);a.review.mkdir(parents=True,exist_ok=True)
    if (a.out/'reconciliation.json').exists():raise FileExistsError('Preserve completed reconciliation')
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths.values()},scope='Existing JSON only; no new inference/reference/training; mathematical sensitivity not a changed quality gate',population_checks=dict(train_test_arrays_disjoint=True,clean_field8_9_match_original_train=True,primary_geometry='pt0',clean_geometries=['rest','tr11','va1'],clean_qualification=data['clean']['dynamic']['va1']['qualified_RGB']),rows=rows)
    (a.out/'reconciliation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (a.review/'all_primary_noise_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(rows=len(rows),population_checks=result['population_checks']),indent=2))
if __name__=='__main__':main()
