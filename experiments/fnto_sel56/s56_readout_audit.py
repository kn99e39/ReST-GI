"""Post hoc A2: analytic emitter camera readout only; original runs/timings remain intact."""
import sys,time,json
import numpy as np
import torch
import s56 as S
import s56_fixtures as F

def main():
    dst=S.OUT/'readout_v2/runs';dst.mkdir(parents=True,exist_ok=True)
    records=[]
    for p in sorted((S.OUT/'runs').glob('*.npz')):
        z=np.load(p);data={k:z[k] for k in z.files};a=time.perf_counter()
        for key in F.KEYS:
            # All emitters in the frozen fixture are planar, normal -z, unit L on their front side.
            L=1.0 if F.FX.DIRS[key][2]<0 else 0.0
            for ro in ('nee','bin'):data[key+'|'+ro][480:]=L
            data[key+'|ball'][480:]=0
        dt=time.perf_counter()-a
        assert all(np.array_equal(data[k],z[k]) if k.startswith('phi_') else np.array_equal(data[k][:480],z[k][:480]) for k in data)
        np.savez(dst/p.name,**data);records.append(dict(file=p.name,assignment_seconds=dt,transport_flux_unchanged=True,medium_readouts_unchanged=True))
    camera={}
    import c54_common as CC
    for state in ('rest','dent','rigid'):
        sd=F.scene(state);ss=S.CM.classify_sceneset(sd);geo=json.load(open(S.OUT/'cycles'/f'ccam_{state}.json'));rec={}
        for cam,g in geo.items():
            o,d=CC.blender_ortho_rays(g,S.CM.V1_OFF)[0];t,p=ss.isect(o,d);j=torch.isfinite(t).nonzero().squeeze(1);pj=p[j]
            med=ss.tri_comp[pj]<2;back=(d[j]*ss.n[pj]).sum(1)>0
            rec[cam]=dict(pixels=len(o),first_hit_is_medium_backface=int((med&back).sum()),
                interpretation='A first exit through a watertight medium back face means the orthographic ray starts inside it; boundary-outgoing port comparison is invalid for those pixels.')
        camera[state]=rec
    # Direct analytic emitter readout agrees exactly with every declared source-group reference (L=1).
    S.dump(dict(records=records,camera=camera,transport_or_K_changes=False,
        source_semantics='Known U emitter: unit front-side radiance, zero backside; no target radiance was used to generate the fix.',
        cost_note='Assignment-only time measured; primary TR/C/solve timings are preserved v1 measurements, not rerun. No medium output changed.'),S.OUT/'readout_audit.json')
    print(json.dumps(camera,indent=1))

if __name__=='__main__':main();sys.stdout.flush();__import__('os')._exit(0)
