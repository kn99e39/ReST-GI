import sys, os, json
import torch
from common import *
from r57_action_oracle import action

def link(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert target.resolve()==source.resolve(), 'Do not overwrite inherited evidence'
    else: target.symlink_to(source)

def prepare():
    records={}
    for name in RG.TRAIN+RG.VAL+DIAGNOSTIC+['te0']:
        source=OLD/'data/geometry'/f'{name}.pt'
        link(source,DATA/'geometry'/source.name)
        records[str(source)]=sha(source)
    for name in RG.TRAIN+RG.VAL:
        source=OLD/'data/training_targets'/f'{name}_r0.pt'
        link(source,DATA/'training_targets'/source.name)
        records[str(source)]=sha(source)
    for name in ['rest']+DIAGNOSTIC:
        for rep in (0,1):
            source=OLD/'data/targets'/f'{name}_r{rep}.pt'
            link(source,DATA/'targets'/source.name)
            records[str(source)]=sha(source)
    link(OLD/'data/lights.pt',DATA/'lights.pt')
    records[str(OLD/'data/lights.pt')]=sha(OLD/'data/lights.pt')
    for name in ('rest','te0'):
        for rep in (0,1):
            source=OLD/'ops'/f'{name}_r{rep}.pt'
            link(source,ROOT/'ops'/source.name)
    for name in PRIMARY:
        path=DATA/'geometry'/f'{name}.pt'
        if not path.exists(): save(RG.serial(geometry(name)),path)
    dump(dict(inherited=records,source='WL57 completed A2, immutable read-only symlinks',
              primary_geometry=CFG['primary_pose'],light_split='96train/32validation/64test; disjoint arrays'),OUT/'data_manifest.json')

def generate():
    fields=torch.load(DATA/'lights.pt')['test']
    for name in PRIMARY:
        mesh,V=RG.mesh(name);fm,VC,parent=RG.KO.refined(mesh,V,4);dom=RG.KO.DomainX(fm,VC)
        for fi in range(64):
            for rep in (0,1):
                for half in (0,1):
                    path=DATA/'halves'/f'{name}_{fi}_{rep}_{half}.pt';receipt=path.with_suffix('.json')
                    if path.exists():
                        if receipt.exists() and json.loads(receipt.read_text())['sha256']==sha(path):continue
                        # Preserve an orphan/partial receipt rather than accepting it or deleting it.
                        import time
                        quarantine=ROOT/'quarantine'/str(time.time_ns());quarantine.mkdir(parents=True)
                        path.rename(quarantine/path.name)
                        if receipt.exists():receipt.rename(quarantine/receipt.name)
                    seed=599000000+fi*100+rep*10+half
                    result=action(dom,fields[fi],32_000_000,seed)
                    assert result['physical_passed'], 'Primary reference failed physical control; preserve log and stop'
                    save(result,path)
                    dump(dict(sha256=sha(path),seed=seed,N=32_000_000,commit=(ROOT/'repo_commit.txt').read_text().strip(),
                              protocol_sha256=sha(HERE/'protocol/persist_v1.json'),physical_passed=True,seconds=result['seconds']),receipt)
                    print('REFERENCE_HALF',name,fi,rep,half,result['seconds'],flush=True)
        for rep in (0,1):
            ys=[]
            for fi in range(64):
                vals=[torch.load(DATA/'halves'/f'{name}_{fi}_{rep}_{half}.pt')['response'][0] for half in (0,1)]
                ys.append(.5*(vals[0]+vals[1]))
            save(dict(test=torch.stack(ys)),DATA/'targets'/f'{name}_r{rep}.pt')
        del dom
        torch.cuda.empty_cache()

if __name__=='__main__':
    {'prepare':prepare,'reference':generate}[sys.argv[1]]()
    sys.stdout.flush();os._exit(0)
