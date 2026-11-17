"""Post-8DNA CK2 static readout of the fixed WL100 near model; no fitting.

Inference uses current CK2 geometry/lights/material features and computes both
physical LOW and learned H itself. References are read only when plotting.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

BASE = Path.home() / 'rest_8dna_ck21b'
SOURCE = Path.home() / 'rest_x93d_ck2' / 'code' / 'experiments'
OUT = BASE / 'results' / 'ecro_ck2_wl100_near_4096spp'
ANCHOR = BASE / 'results' / '8dna_ck2_epoch30_4096spp'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jd(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def render():
    gate = json.loads((ANCHOR / 'matched4096_complete.json').read_text())
    assert gate['epochs_completed'] == 30 and gate['spp_completed'] == 4096
    assert gate['checkpoint_sha256'] == 'd107c982ce9546774acf2c0d575c0f151c482a50af1f6f6a1b92e15c7fafd077'
    protocol = json.loads((BASE/'scripts/ecro_followup.json').read_text())
    assert sha(__file__) == protocol['followup_source_sha256']
    checkpoint = BASE / 'inputs/ecro_wl100_near.pt'
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    for name, digest in protocol['evaluation_source_sha256'].items():
        assert sha(SOURCE/name) == digest, name
    sys.path.insert(0, str(SOURCE/'ecro_local93d/addendum_checker'))
    import checker93d as CK
    import torch
    import model86 as M
    import ecro86 as EC
    _, PT, S, FX, ctx, lights = CK.ctx_of('bx38', str(Path.home()/'rest_x93d_ck2'))
    assert CK.SC == protocol['scene']
    state_dict = torch.load(checkpoint, map_location='cuda')
    model = M.ECRO().cuda()
    model.mat_states = torch.empty_like(state_dict['mat_states'])
    model.load_state_dict(state_dict, strict=True)
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), state_dict[name]), name
    training_material_hash = hashlib.sha256(model.mat_states.cpu().numpy().tobytes()).hexdigest()
    original = np.load(SOURCE/'ecro86/data/mats86.npz')['states']
    current_materials = np.concatenate([original, CK.new_rows()])
    assert np.array_equal(current_materials, ctx.mats.states)
    # Material states are input features, not fitted weights: preserve CK2 S1.
    model.mat_states = torch.as_tensor(current_materials, device='cuda')
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), state_dict[name]), name
    model.eval()
    t0 = time.perf_counter()
    st = EC.build_state(ctx)
    source = ctx.source(lights)
    table, _ = EC.respond(model, st, source)
    assert np.isfinite(table).all()
    np.save(OUT/'response.npy', table)
    del model, state_dict, _
    torch.cuda.empty_cache()
    B = PT.fl(table)
    preparation_seconds = time.perf_counter()-t0
    cam = FX['camera']
    assert cam == CK.SC['camera'] and cam['res'] == 512
    total = {}
    timings = []
    for offset in range(0, 4096, 128):
        acc, timing = ctx.render(lights, 'ER', cam, 512, 128,
                                211300+offset//128, 'ecro', B=B, b=8)
        for key, value in acc.items():
            total[key] = total.get(key, 0) + value
        timings.append(timing)
        jd(OUT/'progress.json', {'status':'rendering','completed_spp':offset+128,
                                'target_spp':4096,'wall_s':time.perf_counter()-t0})
        print('ECRO_SPP', offset+128, flush=True)
    channels = PT.pack(total, 512, 4096)
    assert all(k in channels for k in ['e1','e2','hl'])
    hdr = sum(channels[k].astype(np.float64) for k in ['e1','e2','hl']).astype(np.float32)
    assert hdr.shape == (512,512,3) and np.isfinite(hdr).all()
    np.savez_compressed(OUT/'channels.npz', **channels)
    np.save(OUT/'ecro_ck2_wl100_near_4096spp.npy', hdr)
    PT.mi.Bitmap(hdr).write(str(OUT/'ecro_ck2_wl100_near_4096spp.exr'))
    from PIL import Image
    positive = np.maximum(hdr, 0)
    Image.fromarray(np.rint((positive/(1+positive))**(1/2.2)*255).astype(np.uint8)).save(OUT/'ecro_ck2_wl100_near_4096spp.png')
    record = {'status':'render_complete','spp':4096,'resolution':[512,512],
              'checkpoint_sha256':sha(checkpoint),'learned_parameters_unchanged':True,
              'training_material_buffer_sha256':training_material_hash,
              'evaluation_material_buffer_sha256':hashlib.sha256(current_materials.tobytes()).hexdigest(),
              'reference_used_at_inference':False,'composition':'own physical e1 + e2 + learned hl',
              'protocol':protocol,'preparation_seconds':preparation_seconds,'batch_timings':timings,
              'total_wall_s':time.perf_counter()-t0,
              '8dna_completion_manifest_sha256':sha(ANCHOR/'matched4096_complete.json')}
    jd(OUT/'render_complete.json', record)


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    record = json.loads((OUT/'render_complete.json').read_text())
    ecro = np.load(OUT/'ecro_ck2_wl100_near_4096spp.npy')
    neural = np.load(ANCHOR/'8dna_ck2_epoch30_4096spp.npy')
    root = Path.home()/'rest_x93d_ck2/results'
    refs = sorted((root/'renders').glob('REF_ER_CK_bx38_CK2_T1_r*.npz'))
    assert len(refs) == 4
    reference = []
    for path in refs:
        with np.load(path) as z:
            reference.append(sum(z[k].astype(np.float64) for k in ['e1','e2','h']))
    reference = np.mean(reference, axis=0)
    with np.load(root/'pre/masks_ck2.npz') as z:
        roi = z['ROI_T']
    ys,xs = np.nonzero(roi)
    crop = (slice(max(0,ys.min()-16),min(512,ys.max()+17)),
            slice(max(0,xs.min()-16),min(512,xs.max()+17)))
    error = [np.abs(neural-reference).mean(-1), np.abs(ecro-reference).mean(-1)]
    metrics = {name:{region:float(np.abs(im[m]-reference[m]).sum()/reference[m].sum())
               for region,m in [('frame',np.ones_like(roi)),('reflection',roi)]}
               for name,im in [('8dna',neural),('ecro_wl100_near',ecro)]}
    fig, axes = plt.subplots(2,5,figsize=(18,8.2))
    titles = ['CK2 REF-ER\n4 x 8192 spp; 6 events','8DNA: final 30 epochs\n4096 spp',
              'ECRO WL100 near checkpoint\n4096 spp; own LOW + learned H','|8DNA - REF|','|ECRO - REF|']
    vmax = np.sqrt(max(float(e.max()) for e in error))
    for row in range(2):
        cut = (slice(None),slice(None)) if row == 0 else crop
        for col,image in enumerate([reference,neural,ecro]):
            x = np.maximum(image[cut],0)
            axes[row,col].imshow((x/(1+x))**(1/2.2))
        for col,e in zip([3,4],error):
            mapped = axes[row,col].imshow(np.sqrt(e[cut]),cmap='viridis',vmin=0,vmax=vmax)
        for col,ax in enumerate(axes[row]):
            ax.set_title(titles[col] if row == 0 else 'Fixed reflection crop',fontsize=10)
            ax.axis('off')
    fig.subplots_adjust(left=.015,right=.91,top=.87,bottom=.15,wspace=.08,hspace=.18)
    fig.colorbar(mapped,cax=fig.add_axes([.93,.22,.012,.54]),label='sqrt(mean RGB |HDR difference|), shared scale')
    fig.suptitle('CK2 checkerboard: independent 4096-spp candidate readouts',fontsize=15)
    fig.text(.02,.065,'Original CK2 geometry, S1 ER materials, checker 0.3/0.8, point light and camera. No denoising. Exposure 1, Reinhard, gamma 2.2.\nECRO: fixed WL100 near-mirror-trained weights, evaluated with original CK2 material features (material-distribution transfer; no refit).\n8DNA native collapsed-asset transport differs from the six-event reference. Descriptive static comparison; no dynamic or speed verdict.',fontsize=9)
    name = 'ck2_8dna_epoch30_ecro_wl100_matched4096spp.png'
    fig.savefig(OUT/name,dpi=140);plt.close(fig)
    files = {p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.png','.npy','.npz','.exr']}
    jd(OUT/'complete.json',{**record,'status':'complete','metrics_descriptive_only':metrics,
       'reference_source_hashes':{str(p):sha(p) for p in refs},'files':files})
    print('ECRO_COMPARISON_COMPLETE',name,flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'complete.json').exists():
        if not (OUT/'render_complete.json').exists():
            render()
        figure()
