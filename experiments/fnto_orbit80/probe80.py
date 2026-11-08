"""WL80 non-evidence full-scale probe of the Stage 1 basis fit: sufficient-statistics identity on real targets, FISTA seconds per
(target x iteration) and peak GPU memory for one chunk, Adam A/O step time on the full statistics.  python probe80.py <out.json>"""
import sys,os,json,time
import torch
from common80 import ROOT,environment,load,CFG
RG=environment()
import fit71,family68 as FM,ato74 as AT,basis80 as B
torch.backends.cuda.matmul.allow_tf32=False
tf=load(ROOT/'basis/targets80.pt',mmap=True);S,D,rows=tf['S'],tf['d'],tf['rows'];gs=fit71.geometries(RG);bm=FM.bin_map('cuda');H=FM.Hmat('cuda',torch.float64)
ch=CFG['basis']['chunk'];c=list(range(ch));a,o,_=B.init_logits(0);A,O=(x.cuda() for x in AT.AO(a,o));P=S.shape[2];out={}
torch.cuda.reset_peak_memory_stats();tg=AT.target(S[c].cuda().double(),D[c].cuda().double(),gs['rest']['parent'],bm)
g=torch.Generator(device='cuda').manual_seed(80);T=torch.rand(len(c),P,16,3,device='cuda',dtype=torch.float64,generator=g)/P
with torch.no_grad():v1=B.obj_vec(T,A,O,tg);st=B.stats(T,tg);v2=B.obj_stats(A,O,st,bm,H);v0=AT.objective_m(T,A,O,tg)
out['identity_max_rel']=float(((v1-v2).abs()/v1.abs()).max());out['objective_m_vs_mean']=float(abs(v0-v1.mean())/v0)
torch.cuda.synchronize();t=time.perf_counter();T2,_=AT.fista_T(tg,A,O,T,30);torch.cuda.synchronize();out['fista_seconds_per_target_iter']=(time.perf_counter()-t)/30/len(c)
out['peak_GPU_chunk']=torch.cuda.max_memory_allocated();del tg
st=B.cat_stats([st]*(len(rows)//len(c)+1));a_=a.cuda().requires_grad_(True);o_=o.cuda().requires_grad_(True);opt=torch.optim.Adam([a_,o_],lr=.01)
torch.cuda.synchronize();t=time.perf_counter()
for _ in range(5):opt.zero_grad();L=B.obj_stats(*AT.AO(a_,o_),st,bm,H).sum();L.backward();opt.step()
torch.cuda.synchronize();out['adam_step_seconds_full']=(time.perf_counter()-t)/5;out['targets']=len(rows);out['peak_GPU']=torch.cuda.max_memory_allocated()
R=CFG['basis'];per=out['fista_seconds_per_target_iter']*len(rows)
out['estimated_hours_per_restart']=(R['rounds']*(R['t_iters']*per+R['ao_steps']*out['adam_step_seconds_full'])+R['polish_iters']*per)/3600
json.dump(out,open(sys.argv[1],'w'),indent=1);print('PROBE',json.dumps(out),flush=True);sys.stdout.flush();os._exit(0)
