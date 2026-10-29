"""Summarize immutable local JSON histories and primary/secondary scores with provenance."""
import argparse,hashlib,json,statistics,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'));a=ap.parse_args();r=a.repo
    base=r/'results/fnto_persist59/final';held=json.loads((base/'out/heldout.json').read_text());out={'post_hoc':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r).decode().strip(),'inputs':{},'seeds':{}}
    paths=[base/'out/heldout.json',base/'out/review/noise_sensitivity.json',base/'out/data_manifest.json',r/'experiments/fnto_persist59/protocol/persist_v1.json',r/'experiments/fnto_persist59/train.py',r/'experiments/fnto_rule57/r57_data.py',r/'experiments/fnto_rule57/r57_model.py',r/'experiments/fnto_persist59/runtime.py']
    for seed in (5901,5902):
        p=base/f'out/training/P_{seed}.json';paths.append(p);train=json.loads(p.read_text());h=train['history'];tail=h[-5:]
        poses={}
        for pose,row in held[f'P_{seed}']['poses'].items():
            poses[pose]={'primary':row['primary'],'absolute':row['metrics']['F'],'FS':row['F_S']['max_relative_L1']}
            if 'dynamic' in row:poses[pose].update(dynamic=row['dynamic']['F'],directional=row['directional_pairs']['F'])
        out['seeds'][str(seed)]={'parameters':train['parameters'],'steps':train['steps'],'poses':poses,'history':h,'history_diagnostics':{'initial_val':h[0]['val'],'final_val':h[-1]['val'],'best_logged_val':min(x['val'] for x in h),'best_logged_step':min(h,key=lambda x:x['val'])['step'],'last5_val_range':[min(x['val'] for x in tail),max(x['val'] for x in tail)],'last5_median_delta_weighted_loss_fraction':statistics.median(.25*x['delta']/x['train'] for x in tail),'last5_median_grad_preclip':statistics.median(x['grad_norm'] for x in tail),'caution':'Training values are two-field stochastic snapshots, not full train risk. Validation evaluates only absolute action, not delta or pair contrast.'}}
    for p in paths:out['inputs'][str(p.relative_to(r))]=sha(p)
    out['reference_sensitivity']=json.loads((base/'out/review/noise_sensitivity.json').read_text())
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
