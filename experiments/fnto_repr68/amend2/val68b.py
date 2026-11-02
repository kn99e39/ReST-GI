"""Control: run-to-run nondeterminism of the ORIGINAL path (two identical 1000-step net/all runs), non-evidence."""
import sys,json,torch
sys.path.insert(0,'.');sys.argv=['x']
exec(open('/tmp/nd_val68.py').read().split('# (1)')[0])
OLD['flag']=True;oc=CFG['optimizer']['net'];traj={}
def step(opt,params,cols):
    opt.zero_grad(set_to_none=True);J=pb.loss(cols);J.backward();float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();return J
for rep in ('old_a','old_b'):
    m,params,cols=setup('net','all');opt=torch.optim.Adam(params,lr=oc['lr']);h=[]
    for s in range(1000):
        for q in opt.param_groups:q['lr']=F.cosine(s,1000,oc['lr'],oc['lr_min'])
        J=step(opt,params,cols)
        if s%100==0 or s==999:h.append(float(J))
    traj[rep]=h
print(traj);json.dump(traj,open('/tmp/nd_val68b.json','w'))
