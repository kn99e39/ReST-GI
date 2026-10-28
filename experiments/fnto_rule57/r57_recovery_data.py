"""Prospective A2 dataset: input-importance first-exit actions; old training only.
Immutable original WL57 files are read, never altered. All writes under recovery root.
"""
import os,sys,json,time,shutil
from pathlib import Path
import torch
import r57_geometry as RG
from r57_data import ROOT,OUT,DATA,dump,save,sha,light_fields,aggregate
from r57_action_oracle import action

OLD=Path.home()/'rest_wl57';CFG=json.load(open(os.environ['WL57_CONFIRM_PROTOCOL']))
def link(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.resolve()==Path(source).resolve();return
    target.symlink_to(source)
def prepare():
    DATA.mkdir(parents=True,exist_ok=True);(OUT/'targets').mkdir(parents=True,exist_ok=True)
    owner=DATA/'construction_identity.json';ident=dict(commit=(ROOT/'repo_commit.txt').read_text().strip(),protocol_sha256=sha(os.environ['WL57_CONFIRM_PROTOCOL']))
    if owner.exists():assert json.load(open(owner))==ident,'Partial construction belongs to a different source/config; preserve, do not silently reuse'
    else:dump(ident,owner)
    manifest={}
    for n in RG.TRAIN+RG.VAL:
        gp=OLD/'data/geometry'/f'{n}.pt';link(gp,DATA/'geometry'/f'{n}.pt')
        old=OLD/'data/targets'/f'{n}_r0.pt';p=DATA/'training_targets'/f'{n}_r0.pt'
        if not p.exists():
            obj=torch.load(old);assert {'train','validation'}<=set(obj);save({k:obj[k] for k in ('train','validation')},p)
        data=torch.load(p);assert set(data)=={'train','validation'}
        manifest[n]=dict(original_target=str(old),source_sha256=sha(old),training_sha256=sha(p),keys=list(data))
    # Only for conditional historical WL56 transfer; never in new primary test/training.
    link(OLD/'data/geometry/te0.pt',DATA/'geometry/te0.pt')
    for n in ('rest','te0'):
        for r in (0,1):link(OLD/'ops'/f'{n}_r{r}.pt',ROOT/'ops'/f'{n}_r{r}.pt')
    for name in RG.TEST:
        p=DATA/'geometry'/f'{name}.pt'
        if not p.exists():g=RG.geometry(name);save(RG.serial(g),p);del g;torch.cuda.empty_cache()
    if not (DATA/'lights.pt').exists():
        old=torch.load(OLD/'data/lights.pt');g=RG.geometry('rest',ballistic=False)
        new,meta=light_fields(g,64,CFG['confirmation']['light_seed'],'test')
        # Exact training input reuse, new final test population with fixed generator.
        import hashlib
        hashes=lambda xs:{hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in xs}
        assert not hashes(new)&(hashes(old['train'])|hashes(old['validation'])|hashes(old['test']))
        save(dict(train=old['train'],validation=old['validation'],test=new),DATA/'lights.pt');dump(meta,OUT/'new_light_manifest.json')
    dump(dict(training=manifest,protocol_sha256=sha(os.environ['WL57_CONFIRM_PROTOCOL']),lights_sha256=sha(DATA/'lights.pt'),model_sha256=sha(RG.HERE/'r57_model.py')),OUT/'data_identity.json')

def path(name,field,rep,half):return DATA/'halves'/f'{name}_f{field:02}_r{rep}_h{half}.pt'
def checked(p):
    receipt=p.with_suffix('.json')
    if not p.exists():return False
    assert receipt.exists(),'Complete half file has no receipt; preserve and inspect before resuming'
    d=json.load(open(receipt));assert d['sha256']==sha(p) and d['protocol_sha256']==sha(os.environ['WL57_CONFIRM_PROTOCOL'])
    assert d['commit']==(ROOT/'repo_commit.txt').read_text().strip();return True
def generate(first,last):
    x=torch.load(DATA/'lights.pt')['test']
    for pi,name in enumerate(['rest']+RG.TEST):
        m,V=RG.mesh(name);fm,VC,parent=RG.KO.refined(m,V,4);dom=RG.KO.DomainX(fm,VC,name=name)
        for fi in range(first,last):
            for rep in range(2):
                for half in range(2):
                    p=path(name,fi,rep,half)
                    if checked(p):continue
                    seed=579000000+pi*100000+fi*100+rep*10+half
                    result=action(dom,x[fi],CFG['estimator']['N_per_half'],seed)
                    save(result,p);dump(dict(pose=name,field=fi,rep=rep,half=half,N=result['N'],seed=seed,sha256=sha(p),protocol_sha256=sha(os.environ['WL57_CONFIRM_PROTOCOL']),commit=(ROOT/'repo_commit.txt').read_text().strip(),physical=result['physical_passed'],seconds=result['seconds']),p.with_suffix('.json'))
                    print('HALF',name,fi,rep,half,result['seconds'],result['physical_passed'],flush=True)
                    assert result['physical_passed'],'Reference action physical failure: no adaptive retry'
        del dom;torch.cuda.empty_cache()

def bundle():
    for name in ['rest']+RG.TEST:
        for rep in (0,1):
            ys=[]
            for fi in range(64):
                vals=[]
                for half in (0,1):
                    p=path(name,fi,rep,half);assert checked(p);r=torch.load(p);assert r['physical_passed'];vals.append(r['response'][0])
                ys.append(.5*(vals[0]+vals[1]))
            save({'test':torch.stack(ys)},DATA/'targets'/f'{name}_r{rep}.pt')
    dump(dict(kind='Two independent64M-photon action bundles per pose/field; each is mean of two independent32M halves. Historical training targets kept separately.',protocol_sha256=sha(os.environ['WL57_CONFIRM_PROTOCOL'])),OUT/'bundle_manifest.json')
if __name__=='__main__':
    {'prepare':prepare,'focus':lambda:generate(0,16),'complete':lambda:(generate(16,64),bundle())}[sys.argv[1]]();sys.stdout.flush();os._exit(0)
