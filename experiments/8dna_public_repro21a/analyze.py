"""Summarize raw saved outputs; preserve sampling noise separately from model bias."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def display(x): return np.clip(np.maximum(x,0)**(1/2.2),0,1)
def mse(x,y): return float(np.mean((x-y)**2))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--export',action='store_true'); a=p.parse_args()
    base=Path(a.base); results=base/'results'; protocol=json.loads((base/'scripts'/'protocol.json').read_text())
    report={'protocol':protocol['protocol'],'assets':{},'complete_quality':False,'comparison_limitations':[protocol['scene_mapping_limit'],'RTX 3080 Ti versus paper RTX 3090; timing is descriptive, not identical-hardware reproduction.','Two render replicates estimate Monte Carlo noise but are not a confidence interval.']}
    manifest=[]
    for asset in protocol['assets']:
        entry={}; images={}; metas={}
        for method in ['neural','pt']:
            for rep in [0,1]:
                key=f'{asset}_{method}_r{rep}_{protocol["resolution"]}'
                meta=results/(key+'.json'); image=results/(key+'.npz')
                if not meta.exists() or not image.exists(): continue
                m=json.loads(meta.read_text()); metas[(method,rep)]=m
                with np.load(image) as z:
                    images[(method,rep)]=(z['sum']/m['completed_spp'],z['first128']/128)
                entry[f'{method}_r{rep}_spp']=m['completed_spp']
                if m['completed_spp']>=128:
                    entry[f'{method}_r{rep}_128spp_render_s']=sum(c['wall_s'] for c in m['chunks'][:128//m['arguments']['chunk']])
        if len(images)==4:
            n0,n1=images[('neural',0)][0],images[('neural',1)][0]
            r0,r1=images[('pt',0)][0],images[('pt',1)][0]
            n=(n0+n1)/2; r=(r0+r1)/2
            noise=(mse(n0,n1)+mse(r0,r1))/4
            entry.update(linear_100mse=100*mse(n,r),linear_100noise_floor=100*noise,linear_100cross_mse=100*float(np.mean((n0-r0)*(n1-r1))),display_100mse=100*mse(display(n),display(r)),paper_table1_100mse=protocol['paper_table1_100mse'][asset],linear_ratio_to_paper=mse(n,r)*100/protocol['paper_table1_100mse'][asset],display_ratio_to_paper=mse(display(n),display(r))*100/protocol['paper_table1_100mse'][asset])
            nsec=np.mean([entry[f'neural_r{i}_128spp_render_s'] for i in [0,1]])
            rsec=np.mean([entry[f'pt_r{i}_128spp_render_s'] for i in [0,1]])
            entry.update(neural_128spp_mean_s=float(nsec),pt_128spp_mean_s=float(rsec),pt_over_neural_speedup=float(rsec/nsec),paper_pt_over_neural_speedup=protocol['paper_table2_minutes_pt'][asset]/protocol['paper_table2_minutes_neural'][asset])
            for method in ['neural','pt']:
                # Independent equal-spp differences have expected MSE = 2*variance.
                entry[f'{method}_100variance128_independent_pair']=50*mse(images[(method,0)][1],images[(method,1)][1])
                entry[f'paper_{method}_100variance128']=protocol[f'paper_table2_100variance_{method}'][asset]
            entry['independent_pair_variance_limit']='Unbiased point estimate from two independent 128-spp images; not a confidence interval or high-spp bias verdict.'
            for method in ['neural','pt']:
                v=[]
                for rep in [0,1]:
                    s=metas[(method,rep)]['completed_spp']
                    if s>128:
                        avg,low=images[(method,rep)]; high=(avg*s-low*128)/(s-128)
                        v.append(100*mse(low,high))
                if v: entry[f'{method}_100variance128_vs_disjoint_high']=v
            entry['quality_sample_target_reached']=all(metas[('neural',i)]['completed_spp']>=4096 and metas[('pt',i)]['completed_spp']>=protocol['extra_reference_spp_per_replicate'].get(asset,4096) for i in [0,1])
            entry['precision_diagnostic_noise_over_paper_mse']=100*noise/protocol['paper_table1_100mse'][asset]
            if a.export:
                from PIL import Image,ImageDraw
                scale=max(float(np.quantile(np.abs(n-r),0.99)),1e-9)
                err=np.clip(np.abs(n-r)/scale,0,1)
                panels=np.concatenate([display(r),display(n),err],axis=1)
                canvas=Image.new('RGB',(panels.shape[1],panels.shape[0]+40),'white'); canvas.paste(Image.fromarray((panels*255+0.5).astype('uint8')),(0,40))
                d=ImageDraw.Draw(canvas); d.text((12,12),'Physical PT reference',fill='black'); d.text((812,12),'8DNA released checkpoint',fill='black'); d.text((1612,12),f'|HDR error| / {scale:.4g} (99th percentile)',fill='black')
                path=base/'exports'/f'{asset}_comparison.png'; path.parent.mkdir(exist_ok=True); canvas.save(path)
                manifest.append({'path':str(path.relative_to(base)),'sha256':sha(path),'source_spp':{f'{m}_{i}':metas[(m,i)]['completed_spp'] for m,i in metas},'display':'fixed gamma 1/2.2; HDR errors unmodified in report; error image normalized by labelled 99th percentile'})
        report['assets'][asset]=entry
    report['complete_quality']=all(e.get('quality_sample_target_reached',False) for e in report['assets'].values())
    (base/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if a.export: (base/'exports'/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
