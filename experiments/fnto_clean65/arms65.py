"""Isolated A/B roots: identical frozen WL59 runner; B differs only in train[80:96] labels of rest+12 train dents."""
import os,sys,json,hashlib,time
from pathlib import Path
HOME=Path.home();ROOT=Path(os.environ.get('C65X_ROOT',HOME/'rest_clean65_ext'));ORIG=HOME/'rest_persist_v1';HERE=Path(__file__).resolve().parent
os.environ['P59']=str(ORIG);os.environ['WL57']=str(ORIG)
sys.path.insert(0,str(ORIG/'repo/experiments/fnto_persist59'))
import torch,numpy as np
from common import RG
from runtime import Transport
from r57_data import dump,save,sha
sys.path.insert(0,str(HERE))
from ext65 import override
CFG=json.loads((HERE/'protocol/topup_ab_v2.json').read_text())
FIELDS=list(range(80,96))

def schedule(steps=6000):
    """Independent reconstruction of train.py's arm-independent geometry/field schedule."""
    rng=np.random.default_rng(574001);h=hashlib.sha256();counts=np.zeros(96,int)
    for step in range(steps):
        name=RG.TRAIN[1+step%(len(RG.TRAIN)-1)];ids=rng.integers(0,96,2);h.update(name.encode());h.update(ids.astype(np.int64).tobytes());np.add.at(counts,ids,1)
    return h.hexdigest(),rng.bit_generator.state,counts

def init_hash(seed):
    torch.manual_seed(seed);np.random.seed(seed);m=Transport('P')
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest(),sum(p.numel() for p in m.parameters())

def link(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src,target_is_directory=src.is_dir())

def main():
    q=json.loads((ROOT/'qualification_status.json').read_text());assert q['status']=='READY_FOR_CONTROLLED_TRAINING'
    assert json.loads((ROOT/'fresh/independent_complete.json').read_text())['evaluation_only']
    arms=ROOT/'arms';assert not arms.exists();torch.set_num_threads(8)
    bank_manifest=json.loads((ROOT/'labels_v2_manifest.json').read_text());assert bank_manifest['qualified_for_controlled_intervention']
    commit=(ROOT/'source_commit.txt').read_text().strip();records={}
    for arm in ('A','B'):
        ar=arms/arm;ar.mkdir(parents=True);(ar/'out').mkdir();(ar/'checkpoints').mkdir();(ar/'tmp').mkdir()
        (ar/'out/gate1.json').write_bytes((ORIG/'out/gate1.json').read_bytes());(ar/'repo_commit.txt').write_text(commit+'\n');link(ORIG/'repo',ar/'repo')
    link(ORIG/'data',arms/'A/data')
    data=arms/'B/data';data.mkdir()
    for name in ('geometry','targets','halves','lights.pt'):link(ORIG/'data'/name,data/name)
    tt=data/'training_targets';tt.mkdir()
    for name in RG.VAL:link((ORIG/'data/training_targets'/f'{name}_r0.pt').resolve(),tt/f'{name}_r0.pt')
    changed_fraction={}
    for row in bank_manifest['labels']:
        pose=row['pose'];assert sha(Path(row['path']))==row['sha256']
        bank=torch.load(row['path']);assert bank['fields']==FIELDS and bank['geometry']==pose
        src=ORIG/'data/training_targets'/f'{pose}_r0.pt';orig=torch.load(src);new=override(orig,bank['response'])
        dst=tt/f'{pose}_r0.pt';save(new,dst)
        # Reload from disk: label override isolation on the exact bytes training will read.
        chk=torch.load(dst);assert set(chk)==set(orig)
        assert torch.equal(chk['train'][:80],orig['train'][:80]) and torch.equal(chk['train'][80:],bank['response']) and torch.equal(chk['validation'],orig['validation'])
        assert not torch.equal(chk['train'][80:],orig['train'][80:])
        o=orig['train'][80:].double();n=chk['train'][80:].double()
        changed_fraction[pose]=dict(relative_L2_change=float((n-o).norm()/n.norm()),fields_changed=FIELDS,fields_identical=list(range(80)))
        records[pose]=dict(original=str(src.resolve()),original_sha256=sha(src),B_sha256=sha(dst),bank_sha256=row['sha256'],bytes=dst.stat().st_size);del orig,new,chk,bank
    for name in RG.TRAIN+RG.VAL:
        a=(arms/'A/data/training_targets'/f'{name}_r0.pt');assert a.resolve()==(ORIG/'data/training_targets'/f'{name}_r0.pt').resolve()
        if name in RG.VAL:assert sha(tt/f'{name}_r0.pt')==CFG['input_expected_sha256'][f'data/training_targets/{name}_r0.pt']==sha(a)
    for name in ('geometry','targets','halves','lights.pt'):assert (arms/'B/data'/name).resolve()==(arms/'A/data'/name).resolve()
    digest,state,counts=schedule();i1=init_hash(5901);i2=init_hash(5902);assert init_hash(5901)==i1 and i1[1]==i2[1]==157024
    dump(dict(source_commit=commit,label_files=records,label_change=changed_fraction,schedule_sha256=digest,schedule_final_numpy_state=state,field_draw_counts=counts.tolist(),
        draws_on_replaced_fields=int(counts[80:].sum()),draws_total=int(counts.sum()),init_state_sha256={'5901':i1[0],'5902':i2[0]},parameters=157024,
        runner=dict(train=sha(ORIG/'repo/experiments/fnto_persist59/train.py'),evaluate=sha(ORIG/'repo/experiments/fnto_persist59/evaluate.py'),cost=sha(ORIG/'repo/experiments/fnto_persist59/cost.py'),run_sh=sha(ORIG/'repo/experiments/fnto_persist59/run.sh')),
        evaluation_inputs_identical=True,time=time.time()),ROOT/'arms_manifest.json')
    print('ARMS_READY',digest,flush=True)
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
