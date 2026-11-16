"""WL100 physical dataset generation, reference renders and model lookup readout.

Commands: dev ROOT; freeze ROOT; gen ROOT CASE SPLIT; render ROOT CASE;
          tables ROOT CASE (loads trained weights, never reference targets).
"""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import common100 as C


def context(case,cfg):
    from pt100 import Ctx,PT
    return Ctx(cfg['geom'],case),PT


def gen(root,case,split):
    from pt100 import Ctx,PT
    root=Path(root);rows=json.loads((root/'configs.json').read_text())[case][split]
    spec=C.P['splits'][split]
    ctx=None;t0=time.perf_counter()
    out=root/case/'data'/split;out.mkdir(parents=True,exist_ok=True)
    for cfg in rows:
        i=cfg['index'];fn=out/f'g{i:05d}.npz'
        if fn.exists():continue
        if ctx is None:ctx=Ctx(cfg['geom'],case)
        else:ctx.set_geom(cfg['geom'])
        hit,bn,_=ctx.state_rays();s=[];f2=[];fh=[];b2=[];bh=[]
        for li,ls in enumerate(cfg['lights']):
            light=PT.Lights(ls);s.append(ctx.source(light))
            seed=spec['seed']*100+i*16+li
            a,b=ctx.field_components(light,spec.get('spp_per_field',spec.get('spp_per_half')),seed)
            f2.append(a);fh.append(b)
            if split!='train':
                a,b=ctx.field_components(light,spec['spp_per_half'],seed+8)
                b2.append(a);bh.append(b)
        g=ctx.G
        data={'geom':json.dumps(cfg['geom']),'lights':json.dumps(cfg['lights']),
              'config_hash':C.digest(cfg),'pc':g.PC.astype(np.float32),'pn':g.PN.astype(np.float32),
              'pt':g.PT.astype(np.float32),'parea':g.PAREA.astype(np.float32),'pmat':g.PMAT.astype(np.int16),
              'hit':hit,'dbin':bn,'s':np.stack(s),'f2':np.stack(f2),'fh':np.stack(fh),'mat_states':ctx.mats.states}
        if b2:data.update(f2b=np.stack(b2),fhb=np.stack(bh))
        assert all(np.isfinite(v).all() for v in [*s,*f2,*fh,*b2,*bh]),'STOP_NONFINITE_TARGET'
        np.savez_compressed(fn.with_suffix('.tmp.npz'),**data)
        os.replace(fn.with_suffix('.tmp.npz'),fn)
        print('GEN',case,split,i,'seconds',round(time.perf_counter()-t0,2),flush=True)
    C.jd({'case':case,'split':split,'items':len(rows),'seconds':time.perf_counter()-t0,
          'mat_states_sha256':__import__('hashlib').sha256(C.materials(case).tobytes()).hexdigest(),
          'files':{p.name:C.sha(p) for p in out.glob('g*.npz')}},root/case/f'gen_{split}.json')


def render(root,case):
    root=Path(root);rows=json.loads((root/'configs.json').read_text())[case]['test']
    e=C.P['evaluation'];out=root/case/'renders';out.mkdir(parents=True,exist_ok=True)
    ctx=None
    for index in e['test_indices']:
        cfg=rows[index]
        if ctx is None:ctx,PT=context(case,cfg)
        else:ctx.set_geom(cfg['geom'])
        cam=dict(cfg['camera'],res=e['image_resolution']);ls=PT.Lights(cfg['lights'][0])
        gb=ctx.gbuffer(cam,cam['res'])
        np.savez_compressed(out/f'map_{index}.npz',**gb)
        for half,seed in enumerate(e['reference_seeds']):
            path=out/f'REF_{index}_{half}.npz'
            if path.exists():continue
            acc,timing=ctx.render(ls,'ER',cam,cam['res'],e['reference_spp_per_half'],seed+index*10,'cam',b=32)
            z=PT.pack(acc,cam['res'],e['reference_spp_per_half'])
            np.savez_compressed(path,**z)
            C.jd({'config':cfg,'seed':seed+index*10,'spp':e['reference_spp_per_half'],'timing':timing,
                  'protocol_sha256':C.sha(C.HERE/'protocol.json'),'renderer_sha256':C.sha(C.HERE/'pt100.py'),
                  'scope':'6 optical-domain events; SSS entry-to-first-exit = one event, includes ballistic'},path.with_suffix('.json'))
            print('REFERENCE',case,index,half,timing,flush=True)


def tables(root,case):
    import torch
    import model86 as M
    root=Path(root);e=C.P['evaluation'];rows=json.loads((root/'configs.json').read_text())[case]['test']
    ck=root/case/'model.pt';sd=torch.load(ck,map_location='cuda')
    model=M.ECRO().cuda();model.mat_states=torch.as_tensor(C.materials(case),device='cuda');model.load_state_dict(sd);model.eval()
    ctx=None;out=root/case/'renders';out.mkdir(parents=True,exist_ok=True)
    for index in e['test_indices']:
        cfg=rows[index]
        if ctx is None:ctx,PT=context(case,cfg)
        else:ctx.set_geom(cfg['geom'])
        hit,bn,_=ctx.state_rays();gs=M.GeoState(ctx.G,hit,bn,torch.device('cuda'))
        lights=PT.Lights(cfg['lights'][0]);source=torch.as_tensor(ctx.source(lights),device='cuda').unsqueeze(0)
        with torch.no_grad():
            factors=model.factors(gs)
            single=model.apply(factors,gs,source,use_global=False)
            global_=model.apply(factors,gs,source,use_single=False)
            B=(single+global_)[0].cpu().numpy()
        np.savez_compressed(out/f'table_{index}.npz',B=B,single=single[0].cpu().numpy(),global_=global_[0].cpu().numpy())
        cam=dict(cfg['camera'],res=e['image_resolution'])
        acc,timing=ctx.render(lights,'ER',cam,cam['res'],e['prediction_spp'],e['prediction_seed']+index*10,'ecro',B=PT.fl(B),b=32)
        np.savez_compressed(out/f'PRED_{index}.npz',**PT.pack(acc,cam['res'],e['prediction_spp']))
        C.jd({'checkpoint':str(ck),'checkpoint_sha256':C.sha(ck),'config':cfg,'timing':timing,
              'reference_used_at_inference':False},out/f'PRED_{index}.json')
        print('PREDICTION',case,index,timing,flush=True)


def dev(root):
    """Throwaway physics/visibility samples, never used as training or test data."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    from pt100 import PT
    for case in ['near','sss']:
        cfg=C.config(case,'train',99999);ctx,_=context(case,cfg)
        cam=dict(cfg['camera'],res=128);light=PT.Lights(cfg['lights'][0])
        start=time.perf_counter()
        a,timing=ctx.render(light,'ER',cam,128,32,100001,'cam',b=4)
        z=PT.pack(a,128,32);np.savez_compressed(root/f'dev_{case}.npz',**z)
        C.jd({'case':case,'config':cfg,'timing':timing},root/f'dev_{case}.json')
        print('DEV',case,timing,'wall',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    command,root,*args=sys.argv[1:]
    if command=='freeze':C.freeze(root)
    else:{'gen':gen,'render':render,'tables':tables,'dev':dev}[command](root,*args)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
