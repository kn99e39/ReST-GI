"""WL71C local validation: WL57 fast walker + WL63 tags (unchanged, one 32M chunk per packet as WL71/WL72) vs the streaming engine
on the same packets (independent random streams). Statistical identity per packet (energy z, coarse-cell z^2 for B and Scat),
physical gates, wall time and sampled GPU utilisation.   python validate71c.py <out_dir>"""
import sys,os,json,time,types,threading,subprocess,hashlib
from pathlib import Path
for _m in ('resource','fcntl'):sys.modules.setdefault(_m,types.ModuleType(_m))   # Unix-only imports unused on this path (Windows)
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'experiments/fnto_repr68'));sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('P53_MI_VARIANT','cuda_ad_rgb');os.environ['WL57_WALKER']='fast'
import torch
import common68 as C68
RG=C68.environment(REPO)
import r57_action_oracle as AO,r57_fastwalk as FW
from tag_reference import tagged_action
import engine71c as EN

N=int(os.environ.get("V71C_N","32000000"))
PACKETS=[('rest',2085,0),('rest',2085,7),('rest',2085,15),('rest',2707,3),('rest',2707,9),('rest',1123,5),('rest',2422,12),('rest',1025,1),
         ('pt0',2085,7),('pt0',2441,12),('pt0',2069,4),('pt0',2148,10)]

class GPU:
    def __init__(self):self.s=[];self.stop=False;self.t=threading.Thread(target=self.loop,daemon=True);self.t.start()
    def loop(self):
        while not self.stop:
            try:self.s.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).split()[0]))
            except Exception:pass
            time.sleep(.5)
    def done(self):self.stop=True;self.t.join();return dict(mean=sum(self.s)/max(len(self.s),1),n=len(self.s))

def domain(name):
    mesh,V=RG.mesh(name);fm,VC,parent=RG.KO.refined(mesh,V,4);return RG.KO.DomainX(fm,VC),torch.as_tensor(parent)

def field(P,i,a):
    x=torch.zeros(P,16,3);x[i,a,:]=1.;return x

def coarse(y,parent,bm):
    z=torch.zeros(y.shape[0],16,3,dtype=torch.float64).index_add_(1,bm,y.double());return torch.zeros(240,16,3,dtype=torch.float64).index_add_(0,parent,z)

def compare(o,n,N,parent,bm):
    res={}
    for comp in ('B','Scat'):
        m1,m2=o['component_first'][comp][0].double(),n['component_first'][comp][0].double();s1,s2=o['component_second'][comp][0].double(),n['component_second'][comp][0].double()
        v1=(N*s1-m1.square())/(N-1);v2=(N*s2-m2.square())/(N-1);v1=v1.clamp(min=0);v2=v2.clamp(min=0)
        e1,e2=m1.sum((0,1)),m2.sum((0,1));se=(v1.sum((0,1))+v2.sum((0,1))).sqrt();z=((e1-e2)/se.clamp(min=1e-300)).tolist()
        dm=coarse(m1,parent,bm)-coarse(m2,parent,bm);dv=coarse(v1,parent,bm)+coarse(v2,parent,bm);mask=dv>0
        res[comp]=dict(energy_z=z,coarse_mean_z2=float((dm[mask].square()/dv[mask]).mean()),cells=int(mask.sum()),energy_old=e1.tolist(),energy_new=e2.tolist())
    def ok(c):
        r=res[c];return max(abs(v) for v in r['energy_z'])<=4 and (r['cells']<30 or .5<=r['coarse_mean_z2']<=1.5)
    res['passed']=ok('B') and ok('Scat') and o['physical_passed'] and n['physical_passed']
    return res

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    bm=torch.as_tensor(RG.G.bin_parent(8,4));rows=[];timing={}
    for geo in ('rest','pt0'):
        dom,parent=domain(geo);pk=[p for p in PACKETS if p[0]==geo];fs=[field(dom.P,i,a) for _,i,a in pk]
        os.environ['WL57_ACTION_CHUNK']=str(N);torch.cuda.synchronize();g=GPU();t0=time.perf_counter();old=[]
        for k,f in enumerate(fs):old.append(tagged_action(AO,FW,dom,f,N,71001000+k+(100 if geo=='pt0' else 0)))
        torch.cuda.synchronize();timing[f'{geo}_old']=dict(seconds=time.perf_counter()-t0,packets=len(fs),gpu=g.done(),peak=torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats();g=GPU();t0=time.perf_counter()
        new=EN.Engine(dom,N,seed=71002000+(100 if geo=='pt0' else 0)).run(fs,keys=pk)
        torch.cuda.synchronize();timing[f'{geo}_new']=dict(seconds=time.perf_counter()-t0,packets=len(fs),gpu=g.done(),peak=torch.cuda.max_memory_allocated())
        for p,o,n in zip(pk,old,new):
            c=compare(o,n,N,parent,bm);c.update(packet=list(p),old_seconds=o['seconds'],old_stats={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in o['stats'].items()},
                new_stats={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in n['stats'].items()});rows.append(c);print('PKT',p,c['passed'],[round(x,2) for x in c['Scat']['energy_z']],round(c['Scat']['coarse_mean_z2'],3),round(c['B']['coarse_mean_z2'],3),flush=True)
        print('TIME',geo,{k:v for k,v in timing.items() if k.startswith(geo)},flush=True)
        del dom;torch.cuda.empty_cache()
    passed=all(r['passed'] for r in rows)
    rec=dict(status='PASSED' if passed else 'FAILED',packets=rows,timing=timing,N=N,engine_sha256=hashlib.sha256((Path(__file__).parent/'engine71c.py').read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
             rule='per packet: |energy z| <= 4 (every channel) for B and Scat; 0.5 <= coarse-cell mean z^2 <= 1.5 where >= 30 cells have variance; both physical gates pass')
    (out/'validation.json').write_text(json.dumps(rec,indent=1,default=float));print('VALIDATION',rec['status'],flush=True)

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else str(Path.home()/'rest_walker71c'))
